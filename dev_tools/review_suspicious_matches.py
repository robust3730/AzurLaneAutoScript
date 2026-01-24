"""
Interactive script to review suspicious template matches and approve them for resolution-specific asset creation.

Usage:
    python dev_tools/review_suspicious_matches.py
"""

import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path


def get_suspicious_matches():
    """Get all suspicious match directories."""
    log_dir = Path('log/suspicious_matches')
    if not log_dir.exists():
        return []
    
    matches = []
    for match_dir in sorted(log_dir.iterdir()):
        if match_dir.is_dir():
            metadata_file = match_dir / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                matches.append({
                    'dir': match_dir,
                    'metadata': metadata
                })
    
    return matches


def display_comparison(match_info):
    """Display side-by-side comparison of template and screenshot."""
    match_dir = match_info['dir']
    metadata = match_info['metadata']
    
    template_path = match_dir / 'template.png'
    screenshot_path = match_dir / 'screenshot.png'
    
    if not template_path.exists() or not screenshot_path.exists():
        print(f"  [ERROR] Missing images in {match_dir}")
        return False
    
    template = cv2.imread(str(template_path))
    screenshot = cv2.imread(str(screenshot_path))
    
    if template is None or screenshot is None:
        print(f"  [ERROR] Failed to load images from {match_dir}")
        return False
    
    # Create side-by-side comparison
    h = max(template.shape[0], screenshot.shape[0])
    w_total = template.shape[1] + screenshot.shape[1] + 40  # 40px gap
    
    # Create white background
    comparison = np.ones((h + 60, w_total, 3), dtype=np.uint8) * 255
    
    # Place template on left
    comparison[40:40+template.shape[0], 20:20+template.shape[1]] = template
    
    # Place screenshot on right
    offset_x = template.shape[1] + 40
    comparison[40:40+screenshot.shape[0], offset_x:offset_x+screenshot.shape[1]] = screenshot
    
    # Add labels with better font
    cv2.putText(comparison, 'Template (Original)', (20, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(comparison, 'Screenshot (Matched)', (offset_x, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add similarity scores
    sim_text = f"Standard: {metadata['similarity_standard']:.4f}  |  Retry: {metadata['similarity_retry']:.4f}"
    cv2.putText(comparison, sim_text, (20, h + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Convert BGR to RGB for PIL
    comparison_rgb = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)
    
    # Save to temp file and display with PIL
    from PIL import Image
    import tempfile
    
    temp_path = match_dir / 'comparison_temp.png'
    Image.fromarray(comparison_rgb).save(temp_path)
    
    # Display with system default image viewer (better quality than cv2)
    img = Image.fromarray(comparison_rgb)
    img.show(title=f"Review: {metadata['button_name']}")
    
    return True



def get_resolution_specific_path(original_file, resolution):
    """
    Get resolution-specific asset file path.
    
    Args:
        original_file (str): Original asset file path
        resolution (dict): {'width': int, 'height': int}
        
    Returns:
        Path: Resolution-specific file path, or None if not applicable
    """
    if not original_file:
        return None
    
    width, height = resolution['width'], resolution['height']
    
    # Determine resolution label
    if width == 1280 and height == 720:
        return None  # Default 720p, no change needed
    elif width == 1920 and height == 1080:
        res_label = '1080p'
    elif width == 2560 and height == 1440:
        res_label = '1440p'
    elif width == 3840 and height == 2160:
        res_label = '2160p'
    else:
        res_label = f'{height}p'
    
    # Convert to Path and normalize
    original_path = Path(original_file)
    parts = original_path.parts
    
    # Check if it's an assets path
    if len(parts) < 2 or parts[0] not in ('assets', './assets'):
        return None
    
    # Normalize 'assets' vs './assets'
    if parts[0] == './assets':
        base_parts = parts[:1]  # Keep './assets'
        server_idx = 1
    else:
        base_parts = parts[:1]  # Keep 'assets'
        server_idx = 1
    
    # Insert resolution label after server (e.g., 'cn', 'en', etc.)
    # Structure: assets/cn/1080p/ui/NAV_GENERAL.png
    new_parts = base_parts + parts[server_idx:server_idx+1] + (res_label,) + parts[server_idx+1:]
    
    return Path(*new_parts)


def approve_match(match_info):
    """Approve a match and copy screenshot to resolution-specific asset path."""
    match_dir = match_info['dir']
    metadata = match_info['metadata']
    
    original_file = metadata.get('file_path')
    if not original_file:
        print(f"  [ERROR] No file_path in metadata")
        return False
    
    resolution = metadata.get('resolution')
    if not resolution:
        print(f"  [ERROR] No resolution in metadata")
        return False
    
    area = metadata.get('area')
    if not area:
        print(f"  [ERROR] No area in metadata")
        return False
    
    # Get resolution-specific path
    target_path = get_resolution_specific_path(original_file, resolution)
    if not target_path:
        print(f"  [INFO] Resolution {resolution['width']}x{resolution['height']} is default 720p, no action needed")
        return False
    
    # Create target directory
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load screenshot region
    screenshot_path = match_dir / 'screenshot.png'
    try:
        screenshot_region = cv2.imread(str(screenshot_path))
        if screenshot_region is None:
            print(f"  [ERROR] Failed to load screenshot from {screenshot_path}")
            return False
        
        # Create 1280x720 black canvas
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Place screenshot region at the correct position using area from metadata
        x1, y1 = area['x1'], area['y1']
        x2, y2 = area['x2'], area['y2']
        
        # Get region dimensions
        region_h, region_w = screenshot_region.shape[:2]
        expected_h, expected_w = y2 - y1, x2 - x1
        
        # Verify dimensions match
        if region_h != expected_h or region_w != expected_w:
            print(f"  [WARNING] Region size mismatch: got {region_w}x{region_h}, expected {expected_w}x{expected_h}")
            # Try to resize if needed
            if region_h > 0 and region_w > 0:
                screenshot_region = cv2.resize(screenshot_region, (expected_w, expected_h))
        
        # Place region on canvas
        canvas[y1:y2, x1:x2] = screenshot_region
        
        # Save the masked asset
        cv2.imwrite(str(target_path), canvas)
        print(f"  [SUCCESS] Created 1280x720 masked asset: {target_path}")
        
        # Delete the log directory
        shutil.rmtree(match_dir)
        print(f"  [SUCCESS] Removed log directory: {match_dir}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to create masked asset: {e}")
        import traceback
        traceback.print_exc()
        return False


def reject_match(match_info):
    """Reject a match and delete the log directory."""
    match_dir = match_info['dir']
    try:
        shutil.rmtree(match_dir)
        print(f"  [SUCCESS] Rejected and removed: {match_dir}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to remove directory: {e}")
        return False


def main():
    """Main interactive review loop."""
    print("=" * 80)
    print("Suspicious Match Review Tool")
    print("=" * 80)
    print()
    
    matches = get_suspicious_matches()
    
    if not matches:
        print("No suspicious matches found in log/suspicious_matches/")
        print("Run the game at a non-720p resolution to generate suspicious matches.")
        return
    
    print(f"Found {len(matches)} suspicious match(es) to review.")
    print()
    print("Controls:")
    print("  a - Approve (copy screenshot to resolution-specific asset path)")
    print("  r - Reject (delete log entry)")
    print("  s - Skip (keep for later review)")
    print("  q - Quit")
    print()
    
    for i, match_info in enumerate(matches, 1):
        metadata = match_info['metadata']
        
        print(f"\n[{i}/{len(matches)}] Reviewing: {metadata['button_name']}")
        print(f"  File: {metadata.get('file_path', 'N/A')}")
        print(f"  Resolution: {metadata['resolution']['width']}x{metadata['resolution']['height']}")
        print(f"  Similarity (standard): {metadata['similarity_standard']:.4f}")
        print(f"  Similarity (retry): {metadata['similarity_retry']:.4f}")
        print(f"  Timestamp: {metadata['timestamp']}")
        
        # Display comparison
        if not display_comparison(match_info):
            print("  [WARNING] Failed to display comparison, skipping...")
            continue
        
        # Get user input
        while True:
            choice = input("\n  Action [a/r/s/q]: ").lower().strip()
            
            if choice == 'a':
                approve_match(match_info)
                break
            elif choice == 'r':
                reject_match(match_info)
                break
            elif choice == 's':
                print("  [INFO] Skipped")
                break
            elif choice == 'q':
                print("\nQuitting...")
                return
            else:
                print("  Invalid choice. Please enter 'a', 'r', 's', or 'q'.")
    
    print("\n" + "=" * 80)
    print("Review complete!")
    print("=" * 80)



if __name__ == '__main__':
    main()
