import os
import traceback
import json
from pathlib import Path
from datetime import datetime

import imageio
import cv2
from PIL import ImageDraw

from module.base.decorator import cached_property
from module.base.resource import Resource
from module.base.utils import *
from module.logger import logger
from module.config.server import VALID_SERVER


class Button(Resource):
    _similarity_config_cache = {}

    def __init__(self, area, color, button, file=None, name=None):
        """Initialize a Button instance.

        Args:
            area (dict[tuple], tuple): Area that the button would appear on the image.
                          (upper_left_x, upper_left_y, bottom_right_x, bottom_right_y)
            color (dict[tuple], tuple): Color we expect the area would be.
                           (r, g, b)
            button (dict[tuple], tuple): Area to be click if button appears on the image.
                            (upper_left_x, upper_left_y, bottom_right_x, bottom_right_y)
                            If tuple is empty, this object can be use as a checker.
        Examples:
            BATTLE_PREPARATION = Button(
                area=(1562, 908, 1864, 1003),
                color=(231, 181, 90),
                button=(1562, 908, 1864, 1003)
            )
        """
        self.raw_area = area
        self.raw_color = color
        self.raw_button = button
        self.raw_file = file
        self.raw_name = name

        self._button_offset = None
        self._match_init = False
        self._match_binary_init = False
        self._match_luma_init = False
        self.image = None
        self.image_binary = None
        self.image_luma = None

        if self.file:
            self.resource_add(key=self.file)

    cached = ['area', 'color', '_button', 'file', 'name', 'is_gif']

    @cached_property
    def area(self):
        return self.parse_property(self.raw_area)

    @cached_property
    def color(self):
        return self.parse_property(self.raw_color)

    @cached_property
    def _button(self):
        return self.parse_property(self.raw_button)

    @cached_property
    def file(self):
        return self.parse_property(self.raw_file)

    @cached_property
    def name(self):
        if self.raw_name:
            return self.raw_name
        elif self.file:
            return os.path.splitext(os.path.split(self.file)[1])[0]
        else:
            return 'BUTTON'

    @cached_property
    def is_gif(self):
        if self.file:
            return os.path.splitext(self.file)[1] == '.gif'
        else:
            return False

    def __str__(self):
        return self.name

    __repr__ = __str__

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(self.name)

    def __bool__(self):
        return True

    @property
    def button(self):
        if self._button_offset is None:
            return self._button
        else:
            return self._button_offset

    def appear_on(self, image, threshold=10):
        """Check if the button appears on the image.

        Args:
            image (np.ndarray): Screenshot.
            threshold (int): Default to 10.

        Returns:
            bool: True if button appears on screenshot.
        """
        return color_similar(
            color1=get_color(image, self.area),
            color2=self.color,
            threshold=threshold
        )

    def load_color(self, image):
        """Load color from the specific area of the given image.
        This method is irreversible, this would be only use in some special occasion.

        Args:
            image: Another screenshot.

        Returns:
            tuple: Color (r, g, b).
        """
        self.__dict__['color'] = get_color(image, self.area)
        self.image = crop(image, self.area)
        self.__dict__['is_gif'] = False
        return self.color

    def load_offset(self, button):
        """
        Load offset from another button.

        Args:
            button (Button):
        """
        offset = np.subtract(button.button, button._button)[:2]
        self._button_offset = area_offset(self._button, offset=offset)

    def clear_offset(self):
        self._button_offset = None

    def ensure_template(self):
        """
        Load asset image.
        If needs to call self.match, call this first.
        Supports resolution-specific asset loading with fallback to 720p.
        """
        if not self._match_init:
            # Try to load resolution-specific asset if available
            file_to_load = self.file
            if hasattr(self, '_screenshot_resolution'):
                resolution_specific_file = self._get_resolution_specific_file(
                    self.file, self._screenshot_resolution
                )
                if resolution_specific_file != self.file:
                    logger.info(f'Loading resolution-specific asset: {resolution_specific_file}')
                    file_to_load = resolution_specific_file
            
            if self.is_gif:
                self.image = []
                for image in imageio.mimread(file_to_load):
                    image = image[:, :, :3].copy() if len(image.shape) == 3 else image
                    image = crop(image, self.area)
                    self.image.append(image)
            else:
                self.image = load_image(file_to_load, self.area)
            self._match_init = True

    def ensure_binary_template(self):
        """
        Load asset image.
        If needs to call self.match, call this first.
        """
        if not self._match_binary_init:
            if self.is_gif:
                self.image_binary = []
                for image in self.image:
                    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    _, image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    self.image_binary.append(image_binary)
            else:
                image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                _, self.image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            self._match_binary_init = True

    def ensure_luma_template(self):
        if not self._match_luma_init:
            if self.is_gif:
                self.image_luma = []
                for image in self.image:
                    luma = rgb2luma(image)
                    self.image_luma.append(luma)
            else:
                self.image_luma = rgb2luma(self.image)
            self._match_luma_init = True

    def resource_release(self):
        super().resource_release()
        self.image = None
        self.image_binary = None
        self.image_luma = None
        self._match_init = False
        self._match_binary_init = False
        self._match_luma_init = False

    def _get_resolution_label(self, resolution):
        """
        Get resolution label for asset directory.
        
        Args:
            resolution (tuple): (width, height)
            
        Returns:
            str: Resolution label like '1080p', '1440p', etc.
        """
        width, height = resolution
        if width == 1280 and height == 720:
            return None  # Default 720p, no subdirectory
        elif width == 1920 and height == 1080:
            return '1080p'
        elif width == 2560 and height == 1440:
            return '1440p'
        elif width == 3840 and height == 2160:
            return '2160p'
        else:
            # For other resolutions, use height as label
            return f'{height}p'

    def _get_resolution_specific_file(self, original_file, resolution):
        """
        Get resolution-specific asset file path.
        
        Args:
            original_file (str): Original asset file path
            resolution (tuple): (width, height)
            
        Returns:
            str: Resolution-specific file path if exists, otherwise original path
        """
        if not original_file:
            return original_file
            
        resolution_label = self._get_resolution_label(resolution)
        if not resolution_label:
            return original_file
        
        # Convert to Path and normalize
        original_path = Path(original_file)
        parts = original_path.parts
        
        # Check if it's an assets path
        if len(parts) < 2 or parts[0] not in ('assets', './assets'):
            return original_file
        
        # Normalize 'assets' vs './assets'
        if parts[0] == './assets':
            base_parts = parts[:1]  # Keep './assets'
            server_idx = 1
        else:
            base_parts = parts[:1]  # Keep 'assets'
            server_idx = 1
        
        # Insert resolution label after server (e.g., 'cn', 'en', etc.)
        # Structure: assets/cn/1080p/ui/NAV_GENERAL.png
        new_parts = base_parts + parts[server_idx:server_idx+1] + (resolution_label,) + parts[server_idx+1:]
        resolution_specific_path = Path(*new_parts)
        
        # Check if resolution-specific file exists
        if resolution_specific_path.exists():
            return str(resolution_specific_path)
        else:
            return original_file

    def _get_similarity_override(self, similarity, resolution):
        """
        Get similarity override for this button and resolution.
        
        Args:
            similarity (float): Original similarity.
            resolution (tuple): (width, height)
            
        Returns:
            float: Overridden similarity if found, otherwise original.
        """
        resolution_label = self._get_resolution_label(resolution)
        if not resolution_label or not self.file:
            return similarity
            
        # Structure: assets/{server}/{res_label}/similarity.json
        original_path = Path(self.file)
        parts = original_path.parts
        if len(parts) < 2 or parts[0] not in ('assets', './assets'):
            return similarity
            
        server_idx = 1
        config_path = Path(parts[0]) / parts[server_idx] / resolution_label / 'similarity.json'
        config_path_str = str(config_path)
        
        # Check cache first
        if config_path_str in Button._similarity_config_cache:
            data = Button._similarity_config_cache[config_path_str]
        else:
            data = {}
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    pass
            Button._similarity_config_cache[config_path_str] = data
            
        if self.name in data:
            return float(data[self.name])
                
        return similarity

    def _log_suspicious_match(self, template, screenshot_region, sim_standard, sim_retry, 
                              point, resolution):
        """
        Log suspicious match for manual review.
        Only keeps one log per button name to avoid excessive output.
        
        Args:
            template (np.ndarray): Original template image
            screenshot_region (np.ndarray): Matched region from screenshot
            sim_standard (float): Similarity score from standard matching
            sim_retry (float): Similarity score from retry matching
            point (tuple): Match location (x, y)
            resolution (tuple): Original screenshot resolution (width, height)
        """
        try:
            # Check if a log already exists for this button
            log_base_dir = Path('log') / 'suspicious_matches'
            if log_base_dir.exists():
                # Find and remove existing logs for this button
                for existing_dir in log_base_dir.iterdir():
                    if existing_dir.is_dir() and existing_dir.name.endswith(f'_{self.name}'):
                        # logger.info(f'Skip logging suspicious match for {self.name}')
                        return

            # Create log directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            log_dir = log_base_dir / f'{timestamp}_{self.name}'
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Save template image (convert BGR to RGB)
            template_path = log_dir / 'template.png'
            template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(template_path), template_rgb)
            
            # Save screenshot region (convert BGR to RGB)
            screenshot_path = log_dir / 'screenshot.png'
            screenshot_rgb = cv2.cvtColor(screenshot_region, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(screenshot_path), screenshot_rgb)
            
            # Save metadata
            metadata = {
                'button_name': self.name,
                'file_path': self.file if self.file else None,
                'similarity_standard': float(sim_standard),
                'similarity_retry': float(sim_retry),
                'match_point': {'x': int(point[0]), 'y': int(point[1])},
                'area': {
                    'x1': int(self.area[0]), 'y1': int(self.area[1]),
                    'x2': int(self.area[2]), 'y2': int(self.area[3])
                },
                'resolution': {'width': int(resolution[0]), 'height': int(resolution[1])},
                'timestamp': timestamp
            }
            
            metadata_path = log_dir / 'metadata.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f'Logged suspicious match for {self.name} to {log_dir}')
            
        except Exception as e:
            logger.warning(f'Failed to log suspicious match for {self.name}: {e}')


    def match(self, image, offset=30, similarity=0.85):
        """Detects button by template matching. To Some button, its location may not be static.

        Args:
            image: Screenshot.
            offset (int, tuple): Detection area offset.
            similarity (float): 0-1. Similarity.

        Returns:
            bool.
        """
        self.ensure_template()

        if isinstance(offset, tuple):
            if len(offset) == 2:
                offset = np.array((-offset[0], -offset[1], offset[0], offset[1]))
            else:
                offset = np.array(offset)
        else:
            offset = np.array((-3, -offset, 3, offset))
        image = crop(image, offset + self.area, copy=False)

        # Apply similarity override if available
        if hasattr(self, '_screenshot_resolution'):
            similarity = self._get_similarity_override(similarity, self._screenshot_resolution)

        if self.is_gif:
            for template in self.image:
                res = cv2.matchTemplate(template, image, cv2.TM_CCOEFF_NORMED)
                _, sim, _, point = cv2.minMaxLoc(res)
                self._button_offset = area_offset(self._button, offset[:2] + np.array(point))
                if sim > similarity:
                    return True
            return False
        else:
            res = cv2.matchTemplate(self.image, image, cv2.TM_CCOEFF_NORMED)
            _, sim, _, point = cv2.minMaxLoc(res)
            sim_standard = sim  # Store original similarity for logging

            # suspicious match detection
            # If match failed, try cropping template edges (1px) and Gaussian blur
            # This handles cases where template/screenshot has slight positional shift or artifacts
            h, w = self.image.shape[:2]
            if sim < similarity and (h > 4 and w > 4):
                template_retry = self.image[1:-1, 1:-1]
                template_retry = cv2.GaussianBlur(template_retry, (3, 3), 0)
                image_retry = cv2.GaussianBlur(image, (3, 3), 0)
                res_retry = cv2.matchTemplate(template_retry, image_retry, cv2.TM_CCOEFF_NORMED)
                _, sim_retry, _, point_retry = cv2.minMaxLoc(res_retry)
                
                # Retry is only for suspicious match detection, not to override standard matching
                # If retry succeeds where standard failed, log it for manual review
                if sim_retry >= similarity - 0.05:
                    # Adjust point: cropped template starts at (1,1) of original template
                    # So if we found it at (x,y), the original template would be at (x-1, y-1)
                    point_retry_adjusted = (point_retry[0] - 1, point_retry[1] - 1)
                    
                    try:
                        # Get resolution from device (if available)
                        resolution = getattr(self, '_screenshot_resolution', (1280, 720))
                        
                        new_template_region = image[point_retry_adjusted[1]:point_retry_adjusted[1] + h, 
                                                    point_retry_adjusted[0]:point_retry_adjusted[0] + w]
                        if new_template_region.shape == self.image.shape:
                            self._log_suspicious_match(
                                template=self.image,
                                screenshot_region=new_template_region,
                                sim_standard=sim_standard,
                                sim_retry=sim_retry,
                                point=point_retry_adjusted,
                                resolution=resolution
                            )
                    except Exception as e:
                        logger.warning(f'Failed to log suspicious match: {e}')

            self._button_offset = area_offset(self._button, offset[:2] + np.array(point))
            return sim > similarity



    def match_binary(self, image, offset=30, similarity=0.85):
        """Detects button by template matching. To Some button, its location may not be static.
           This method will apply template matching under binarization.

        Args:
            image: Screenshot.
            offset (int, tuple): Detection area offset.
            similarity (float): 0-1. Similarity.

        Returns:
            bool.
        """
        self.ensure_template()
        self.ensure_binary_template()

        if isinstance(offset, tuple):
            if len(offset) == 2:
                offset = np.array((-offset[0], -offset[1], offset[0], offset[1]))
            else:
                offset = np.array(offset)
        else:
            offset = np.array((-3, -offset, 3, offset))
        image = crop(image, offset + self.area, copy=False)

        if self.is_gif:
            for template in self.image_binary:
                # graying
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # binarization
                _, image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                # template matching
                res = cv2.matchTemplate(template, image_binary, cv2.TM_CCOEFF_NORMED)
                _, sim, _, point = cv2.minMaxLoc(res)
                self._button_offset = area_offset(self._button, offset[:2] + np.array(point))
                if sim > similarity:
                    return True
            return False
        else:
            # graying
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # binarization
            _, image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # template matching
            res = cv2.matchTemplate(self.image_binary, image_binary, cv2.TM_CCOEFF_NORMED)
            _, sim, _, point = cv2.minMaxLoc(res)
            self._button_offset = area_offset(self._button, offset[:2] + np.array(point))
            return sim > similarity

    def match_luma(self, image, offset=30, similarity=0.85):
        """
        Detects button by template matching under Y channel (Luminance)

        Args:
            image: Screenshot.
            offset (int, tuple): Detection area offset.
            similarity (float): 0-1. Similarity.

        Returns:
            bool.
        """
        self.ensure_template()
        self.ensure_luma_template()

        if isinstance(offset, tuple):
            if len(offset) == 2:
                offset = np.array((-offset[0], -offset[1], offset[0], offset[1]))
            else:
                offset = np.array(offset)
        else:
            offset = np.array((-3, -offset, 3, offset))
        image = crop(image, offset + self.area, copy=False)

        if self.is_gif:
            image_luma = rgb2luma(image)
            for template in self.image_luma:
                res = cv2.matchTemplate(template, image_luma, cv2.TM_CCOEFF_NORMED)
                _, sim, _, point = cv2.minMaxLoc(res)
                self._button_offset = area_offset(self._button, offset[:2] + np.array(point))
                if sim > similarity:
                    return True
        else:
            image_luma = rgb2luma(image)
            res = cv2.matchTemplate(self.image_luma, image_luma, cv2.TM_CCOEFF_NORMED)
            _, sim, _, point = cv2.minMaxLoc(res)
            self._button_offset = area_offset(self._button, offset[:2] + np.array(point))
            return sim > similarity

    def match_template_color(self, image, offset=(20, 20), similarity=0.85, threshold=30):
        """
        Template match first, color match then

        Args:
            image: Screenshot.
            offset (int, tuple): Detection area offset.
            similarity (float): 0-1.
            threshold (int): Default to 30.

        Returns:
            bool.
        """
        if self.match_luma(image, offset=offset, similarity=similarity):
            diff = np.subtract(self.button, self._button)[:2]
            area = area_offset(self.area, offset=diff)
            color = get_color(image, area)
            return color_similar(color1=color, color2=self.color, threshold=threshold)
        else:
            return False

    def crop(self, area, image=None, name=None):
        """
        Get a new button by relative coordinates.

        Args:
            area (tuple):
            image (np.ndarray): Screenshot. If provided, load color and image from it.
            name (str):

        Returns:
            Button:
        """
        if name is None:
            name = self.name
        new_area = area_offset(area, offset=self.area[:2])
        new_button = area_offset(area, offset=self.button[:2])
        button = Button(area=new_area, color=self.color, button=new_button, file=self.file, name=name)
        if image is not None:
            button.load_color(image)
        return button

    def move(self, vector, image=None, name=None):
        """
        Move button.

        Args:
            vector (tuple):
            image (np.ndarray): Screenshot. If provided, load color and image from it.
            name (str):

        Returns:
            Button:
        """
        if name is None:
            name = self.name
        new_area = area_offset(self.area, offset=vector)
        new_button = area_offset(self.button, offset=vector)
        button = Button(area=new_area, color=self.color, button=new_button, file=self.file, name=name)
        if image is not None:
            button.load_color(image)
        return button

    def split_server(self):
        """
        Split into 4 server specific buttons.

        Returns:
            dict[str, Button]:
        """
        out = {}
        for s in VALID_SERVER:
            out[s] = Button(
                area=self.parse_property(self.raw_area, s),
                color=self.parse_property(self.raw_color, s),
                button=self.parse_property(self.raw_button, s),
                file=self.parse_property(self.raw_file, s),
                name=self.name
            )
        return out


class ButtonGrid:
    def __init__(self, origin, delta, button_shape, grid_shape, name=None):
        self.origin = np.array(origin)
        self.delta = np.array(delta)
        self.button_shape = np.array(button_shape)
        self.grid_shape = np.array(grid_shape)
        if name:
            self._name = name
        else:
            (filename, line_number, function_name, text) = traceback.extract_stack()[-2]
            self._name = text[:text.find('=')].strip()

    def __getitem__(self, item):
        base = np.round(np.array(item) * self.delta + self.origin).astype(int)
        area = tuple(np.append(base, base + self.button_shape))
        return Button(area=area, color=(), button=area, name='%s_%s_%s' % (self._name, item[0], item[1]))

    def generate(self):
        for y in range(self.grid_shape[1]):
            for x in range(self.grid_shape[0]):
                yield x, y, self[x, y]

    @cached_property
    def buttons(self):
        return list([button for _, _, button in self.generate()])

    def crop(self, area, name=None):
        """
        Args:
            area (tuple): Area related to self.origin
            name (str): Name of the new ButtonGrid instance.

        Returns:
            ButtonGrid:
        """
        if name is None:
            name = self._name
        origin = self.origin + area[:2]
        button_shape = np.subtract(area[2:], area[:2])
        return ButtonGrid(
            origin=origin, delta=self.delta, button_shape=button_shape, grid_shape=self.grid_shape, name=name)

    def move(self, vector, name=None):
        """
        Args:
            vector (tuple): Move vector.
            name (str): Name of the new ButtonGrid instance.

        Returns:
            ButtonGrid:
        """
        if name is None:
            name = self._name
        origin = self.origin + vector
        return ButtonGrid(
            origin=origin, delta=self.delta, button_shape=self.button_shape, grid_shape=self.grid_shape, name=name)

    def gen_mask(self):
        """
        Generate a mask image to display this ButtonGrid object for debugging.

        Returns:
            PIL.Image.Image: Area in white, background in black.
        """
        image = Image.new("RGB", (1280, 720), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        for button in self.buttons:
            draw.rectangle((button.area[:2], button.button[2:]), fill=(255, 255, 255), outline=None)
        return image

    def show_mask(self):
        self.gen_mask().show()

    def save_mask(self):
        """
        Save mask to {name}.png
        """
        self.gen_mask().save(f'{self._name}.png')
