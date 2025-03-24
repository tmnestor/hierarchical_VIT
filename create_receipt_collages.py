import os
from pathlib import Path
import random
import argparse
import math
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter
from tqdm import tqdm
from config import get_config

def create_receipt_collage(image_paths, canvas_size=(1600, 1200), receipt_count=None, 
                        bg_color=(245, 245, 245), realistic=True):
    """
    Create a collage of receipts with realistic appearance.
    
    Args:
        image_paths: List of paths to receipt images
        canvas_size: Size of the output canvas (width, height)
        receipt_count: Specific number of receipts to include (or None for random)
        bg_color: Background color of the canvas
        realistic: If True, make the receipts blend with the background
        
    Returns:
        collage_img: PIL Image of the collage
        actual_count: Number of receipts in the collage
    """
    # Create a canvas with background color (slightly off-white for realism)
    canvas = Image.new('RGB', canvas_size, color=bg_color)
    
    # If receipt_count is None, it should be set by the caller
    # based on the class distribution from config
    
    # For 0 receipts, add realistic table/surface elements instead of plain background
    if receipt_count == 0:
        # Add texture/pattern to simulate a real surface
        add_table_texture(canvas)
        
        # Add random table items (not receipts) to create visual complexity
        add_table_elements(canvas)
        
        return canvas, 0
    
    # Randomly select receipt images
    if receipt_count > len(image_paths):
        receipt_count = len(image_paths)
    selected_images = random.sample(image_paths, receipt_count)
    
    # Define a grid for placing receipts
    # Use a 3x2 grid for up to 6 receipts (sufficient for the project's 0-5 receipt classes)
    grid_columns = 3
    grid_rows = 2
    
    # Calculate cell dimensions
    cell_width = canvas_size[0] // grid_columns
    cell_height = canvas_size[1] // grid_rows
    
    # Keep track of which grid cells are used
    grid_used = [[False for _ in range(grid_columns)] for _ in range(grid_rows)]
    
    # Function to get unused grid cell
    def get_unused_cell():
        unused_cells = [(r, c) for r in range(grid_rows) for c in range(grid_columns) 
                        if not grid_used[r][c]]
        if not unused_cells:
            return None
        return random.choice(unused_cells)
    
    # Place each receipt
    actual_count = 0
    
    for img_path in selected_images:
        try:
            # Get an unused cell
            cell = get_unused_cell()
            if cell is None:
                break  # No more cells available
                
            row, col = cell
            grid_used[row][col] = True
            
            # Calculate the cell boundaries
            cell_x = col * cell_width
            cell_y = row * cell_height
            
            # Load and prepare the receipt
            receipt = Image.open(img_path).convert('RGB')
            
            # Make receipts look like white paper on a colored background
            if realistic:
                try:
                    # Detect the dark rectangular background of the receipt
                    # Create a mask where dark pixels are white and light pixels are black
                    dark_mask = Image.new('L', receipt.size, 0)  # Start with black
                    
                    # Find dark pixels (the rectangle surrounding the receipt)
                    for x in range(receipt.width):
                        for y in range(receipt.height):
                            pixel = receipt.getpixel((x, y))
                            avg = sum(pixel) // 3
                            if avg < 100:  # Very dark pixels - the rectangle boundary
                                dark_mask.putpixel((x, y), 255)  # Mark as white in mask
                    
                    # Use the dark_mask to replace the black rectangle with background color
                    receipt_realistic = receipt.copy()
                    for x in range(receipt.width):
                        for y in range(receipt.height):
                            if dark_mask.getpixel((x, y)) > 200:  # It's part of the black rectangle
                                receipt_realistic.putpixel((x, y), bg_color)  # Replace with background color
                    
                    receipt = receipt_realistic
                except Exception as e:
                    print(f"Error processing receipt for realism: {e}")
            
            # Resize to fit within the cell (with margin)
            margin = 20  # pixels margin
            max_width = cell_width - 2 * margin
            max_height = cell_height - 2 * margin
            
            # Calculate scale to fit within max dimensions
            receipt_width, receipt_height = receipt.size
            width_scale = max_width / receipt_width
            height_scale = max_height / receipt_height
            scale = min(width_scale, height_scale)
            
            # If scale > 1, don't enlarge the image
            if scale > 1:
                scale = 1
                
            new_width = int(receipt_width * scale)
            new_height = int(receipt_height * scale)
            receipt = receipt.resize((new_width, new_height), Image.LANCZOS)
            
            # Apply a slight rotation (±10 degrees)
            angle = random.uniform(-10, 10)
            receipt = receipt.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=bg_color)
            
            # Calculate random position within the cell (centered with slight variation)
            cell_center_x = cell_x + cell_width // 2
            cell_center_y = cell_y + cell_height // 2
            
            # Add slight random offset for natural look 
            max_offset = min(20, (cell_width - receipt.width) // 2, (cell_height - receipt.height) // 2)
            max_offset = max(0, max_offset)  # Ensure it's not negative
            offset_x = random.randint(-max_offset, max_offset) if max_offset > 0 else 0
            offset_y = random.randint(-max_offset, max_offset) if max_offset > 0 else 0
            
            # Calculate final position
            paste_x = cell_center_x - receipt.width // 2 + offset_x
            paste_y = cell_center_y - receipt.height // 2 + offset_y
            
            # Add subtle shadow to create depth
            if realistic:
                shadow_offset = 3
                shadow_strength = 30
                shadow_color = (max(0, bg_color[0]-shadow_strength),
                               max(0, bg_color[1]-shadow_strength),
                               max(0, bg_color[2]-shadow_strength))
                
                # Create shadow
                shadow = Image.new('RGB', receipt.size, shadow_color)
                
                # Paste shadow with offset
                canvas.paste(shadow, (paste_x+shadow_offset, paste_y+shadow_offset))
            
            # Paste the receipt onto the canvas
            canvas.paste(receipt, (paste_x, paste_y))
            actual_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    return canvas, actual_count

def add_table_texture(canvas):
    """
    Add realistic table/surface texture to make empty scenes more challenging.
    
    Args:
        canvas: PIL Image to add texture to
    """
    width, height = canvas.size
    
    # Choose a texture style randomly
    texture_style = random.choice(['wood', 'marble', 'fabric', 'concrete', 'noise'])
    
    if texture_style == 'wood':
        # Wood grain texture
        for y in range(height):
            # Create horizontal wood grain with slight variation
            color_offset = random.randint(-15, 15)
            line_color = (
                max(0, min(255, 220 + color_offset + random.randint(-5, 5))),
                max(0, min(255, 190 + color_offset + random.randint(-5, 5))),
                max(0, min(255, 160 + color_offset + random.randint(-5, 5)))
            )
            
            # Periodically add darker grain lines
            if random.random() < 0.05:
                line_color = (
                    max(0, line_color[0] - random.randint(30, 50)),
                    max(0, line_color[1] - random.randint(30, 50)),
                    max(0, line_color[2] - random.randint(30, 50))
                )
            
            for x in range(width):
                if random.random() < 0.7:  # Maintain some continuity
                    canvas.putpixel((x, y), line_color)
    
    elif texture_style == 'marble':
        # Marble-like texture
        draw = ImageDraw.Draw(canvas)
        
        # Base marble coloration (slight variations of white/gray)
        for i in range(20):
            # Create random flowing curves
            points = []
            x, y = random.randint(0, width), random.randint(0, height)
            for _ in range(random.randint(5, 15)):
                x += random.randint(-100, 100)
                y += random.randint(-100, 100)
                points.append((x, y))
            
            # Draw curved lines with random transparency
            color = (
                random.randint(200, 240),
                random.randint(200, 240),
                random.randint(200, 240)
            )
            draw.line(points, fill=color, width=random.randint(1, 10))
        
        # Apply blur to soften the marble effect
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=5))
    
    elif texture_style == 'fabric':
        # Fabric/cloth texture with subtle pattern
        for y in range(height):
            for x in range(width):
                # Create subtle grid pattern
                pattern_val = (math.sin(x/10) * math.sin(y/10)) * 10
                color_offset = int(pattern_val)
                
                # Base color with subtle variation
                base_color = canvas.getpixel((x, y))
                new_color = (
                    max(0, min(255, base_color[0] + color_offset)),
                    max(0, min(255, base_color[1] + color_offset)),
                    max(0, min(255, base_color[2] + color_offset))
                )
                canvas.putpixel((x, y), new_color)
    
    elif texture_style == 'concrete':
        # Concrete/stone texture
        for y in range(height):
            for x in range(width):
                if x % 2 == 0 and y % 2 == 0:  # Optimize by calculating fewer pixels
                    # Create noise pattern
                    noise = random.randint(-15, 15)
                    color = (
                        max(0, min(255, 200 + noise)),
                        max(0, min(255, 200 + noise)),
                        max(0, min(255, 200 + noise))
                    )
                    
                    # Set pixel and neighbors for efficiency
                    try:
                        canvas.putpixel((x, y), color)
                        canvas.putpixel((x+1, y), color)
                        canvas.putpixel((x, y+1), color)
                        canvas.putpixel((x+1, y+1), color)
                    except IndexError:
                        pass
    
    else:  # noise
        # Random noise texture
        for y in range(0, height, 2):  # Step by 2 for efficiency
            for x in range(0, width, 2):
                # Create random noise
                noise = random.randint(-10, 10)
                color = (
                    max(0, min(255, 245 + noise)),
                    max(0, min(255, 245 + noise)),
                    max(0, min(255, 245 + noise))
                )
                
                # Set pixel and neighbors
                try:
                    canvas.putpixel((x, y), color)
                    canvas.putpixel((x+1, y), color)
                    canvas.putpixel((x, y+1), color)
                    canvas.putpixel((x+1, y+1), color)
                except IndexError:
                    pass
    
    # Add some dust/scratches/imperfections regardless of texture
    for _ in range(random.randint(50, 200)):
        x, y = random.randint(0, width-1), random.randint(0, height-1)
        size = random.randint(1, 5)
        color_shift = random.randint(-30, 30)
        
        # Get original color and adjust
        orig_color = canvas.getpixel((x, y))
        mark_color = (
            max(0, min(255, orig_color[0] + color_shift)),
            max(0, min(255, orig_color[1] + color_shift)),
            max(0, min(255, orig_color[2] + color_shift))
        )
        
        # Draw the imperfection
        for dx in range(-size, size+1):
            for dy in range(-size, size+1):
                if dx*dx + dy*dy <= size*size:  # Circular shape
                    try:
                        canvas.putpixel((x+dx, y+dy), mark_color)
                    except IndexError:
                        pass

def add_table_elements(canvas):
    """
    Add random non-receipt items to empty table scenes to increase complexity.
    These could be coffee cups, pens, phones, etc. represented as simple shapes.
    
    IMPORTANT: This version avoids adding any rectangular shapes that could
    be confused with receipts in the 0-receipt case.
    
    Args:
        canvas: PIL Image to add elements to
    """
    width, height = canvas.size
    draw = ImageDraw.Draw(canvas)
    
    # Determine number of items to add (1-5)
    num_items = random.randint(1, 5)
    
    for _ in range(num_items):
        # Randomly choose item type - NO RECTANGULAR OBJECTS!
        item_type = random.choice(['pen', 'cup', 'keys', 'coffee_stain', 'circle', 'shadow'])
        
        # Random position (away from edges)
        x = random.randint(width//10, width*9//10)
        y = random.randint(height//10, height*9//10)
        
        if item_type == 'pen':
            # Draw a pen/pencil (thin line)
            pen_length = random.randint(80, 150)
            pen_width = random.randint(2, 6)  # Thinner than a receipt
            angle = random.uniform(0, 360)
            
            # Calculate endpoints
            dx = pen_length * math.cos(math.radians(angle)) / 2
            dy = pen_length * math.sin(math.radians(angle)) / 2
            
            # Draw with random color
            color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            draw.line([(x-dx, y-dy), (x+dx, y+dy)], fill=color, width=pen_width)
        
        elif item_type == 'cup':
            # Draw a coffee cup (circle)
            cup_radius = random.randint(20, 40)
            cup_color = (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255))
            
            # Cup body (circle or ellipse)
            draw.ellipse([(x-cup_radius, y-cup_radius), (x+cup_radius, y+cup_radius)], 
                         outline=(100, 100, 100), fill=cup_color)
            
            # Cup handle
            handle_x = x + cup_radius
            draw.arc([(handle_x, y-cup_radius//2), (handle_x+cup_radius//2, y+cup_radius//2)], 
                     0, 180, fill=(100, 100, 100), width=3)
        
        elif item_type == 'keys':
            # Draw key ring with keys
            center_color = (180, 180, 180)  # Silver/gray
            draw.ellipse([(x-10, y-10), (x+10, y+10)], outline=(100, 100, 100), fill=center_color)
            
            # Add several keys
            num_keys = random.randint(2, 5)
            for i in range(num_keys):
                angle = i * (360 / num_keys)
                key_length = random.randint(20, 40)
                dx = key_length * math.cos(math.radians(angle))
                dy = key_length * math.sin(math.radians(angle))
                
                # Draw key as a line with small circle at end (no rectangles)
                draw.line([(x, y), (x+dx, y+dy)], fill=center_color, width=2)
                draw.ellipse([(x+dx-5, y+dy-5), (x+dx+5, y+dy+5)], fill=center_color)
        
        elif item_type == 'coffee_stain':
            # Coffee stain (irregular brown circle)
            stain_radius = random.randint(15, 40)
            stain_color = (
                random.randint(150, 190),  # R - brownish
                random.randint(100, 140),  # G
                random.randint(50, 90)     # B
            )
            
            # Create irregular stain shape
            points = []
            num_points = random.randint(8, 15)
            for i in range(num_points):
                angle = i * (360 / num_points)
                radius = stain_radius * random.uniform(0.7, 1.3)
                px = x + radius * math.cos(math.radians(angle))
                py = y + radius * math.sin(math.radians(angle))
                points.append((px, py))
            
            draw.polygon(points, fill=stain_color)
            
            # Add some lighter "splash" circles around it
            num_splashes = random.randint(2, 6)
            for _ in range(num_splashes):
                splash_angle = random.uniform(0, 360)
                splash_dist = stain_radius * random.uniform(0.8, 1.5)
                splash_x = x + splash_dist * math.cos(math.radians(splash_angle))
                splash_y = y + splash_dist * math.sin(math.radians(splash_angle))
                splash_size = random.randint(2, 8)
                
                draw.ellipse([(splash_x-splash_size, splash_y-splash_size), 
                             (splash_x+splash_size, splash_y+splash_size)], 
                             fill=stain_color)
        
        elif item_type == 'circle':
            # Simple circle - could be a coaster, coin, etc.
            circle_radius = random.randint(10, 30)
            circle_color = (
                random.randint(150, 250),
                random.randint(150, 250),
                random.randint(150, 250)
            )
            
            draw.ellipse([(x-circle_radius, y-circle_radius), 
                         (x+circle_radius, y+circle_radius)], 
                         outline=(100, 100, 100), fill=circle_color)
        
        else:  # shadow or blob
            # Random shadow/blob
            shadow_radius = random.randint(15, 50)
            shadow_color = (
                random.randint(180, 220),
                random.randint(180, 220),
                random.randint(180, 220)
            )
            
            # Create an irregular shape
            points = []
            num_points = random.randint(5, 10)
            for i in range(num_points):
                angle = i * (360 / num_points)
                radius = shadow_radius * random.uniform(0.7, 1.3)
                px = x + radius * math.cos(math.radians(angle))
                py = y + radius * math.sin(math.radians(angle))
                points.append((px, py))
            
            draw.polygon(points, fill=shadow_color)
    
    # Optional: Apply slight blur to make elements blend with the background
    if random.random() < 0.7:
        return canvas.filter(ImageFilter.GaussianBlur(radius=1))
    
    return canvas

def main():
    """
    Generate collages of receipts for training vision transformer models.
    
    This script creates synthetic training data by placing receipts on background canvases.
    It fully integrates with the config system to ensure class distribution consistency
    across the entire codebase. The number of receipts (0-5) follows the exact
    distribution specified in the configuration.
    
    The class distribution can be specified via:
    1. Command line with --count_probs
    2. Configuration file with --config
    3. Default from the global config system
    
    Receipt counts are selected according to the probability distribution,
    and the actual distribution is reported at the end of generation.
    
    IMPORTANT: This version generates more challenging and realistic 0-receipt images
    by adding table textures and non-receipt objects rather than using plain backgrounds.
    """
    parser = argparse.ArgumentParser(description="Create receipt collages for training vision transformer models")
    parser.add_argument("--input_dir", default="raw_data", 
                        help="Directory containing receipt images")
    parser.add_argument("--output_dir", default="receipts", 
                        help="Directory to save collage images")
    parser.add_argument("--num_collages", type=int, default=300,
                        help="Number of collages to create")
    parser.add_argument("--canvas_width", type=int, default=1600,
                        help="Width of the collage canvas")
    parser.add_argument("--canvas_height", type=int, default=1200,
                        help="Height of the collage canvas")
    parser.add_argument("--count_probs", type=str, 
                      help="Comma-separated probabilities for 0,1,2,3,4,5 receipts (overrides config)")
    parser.add_argument("--realistic", action="store_true", default=True,
                      help="Make receipts blend with background for more realistic appearance")
    parser.add_argument("--bg_color", type=str, default="245,245,245",
                      help="Background color in RGB format (e.g., '245,245,245' for light gray)")
    parser.add_argument("--config", help="Path to configuration JSON file")
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config_path = Path(args.config) if args.config else None
    
    # Load configuration
    config = get_config()
    if config_path:
        if not config_path.exists():
            print(f"Warning: Configuration file not found: {config_path}")
        else:
            config.load_from_file(config_path, silent=False)  # Explicitly show this load
    
    # Parse probability distribution (command line args override config)
    if args.count_probs:
        try:
            # Parse probabilities from command line
            count_probs = [float(p) for p in args.count_probs.split(',')]
            
            # Normalize to ensure they sum to 1
            prob_sum = sum(count_probs)
            if prob_sum <= 0:
                raise ValueError("Probabilities must sum to a positive value")
            count_probs = [p / prob_sum for p in count_probs]
            print(f"Using receipt count distribution from command line: {count_probs}")
            
            # Update config with the new distribution
            if len(count_probs) == len(config.class_distribution):
                config.update_class_distribution(count_probs)
            else:
                print(f"Warning: Provided distribution has {len(count_probs)} values, " 
                      f"but configuration expects {len(config.class_distribution)}. Using provided values.")
        except (ValueError, AttributeError) as e:
            print(f"Warning: Invalid probability format in command line: {e}")
            count_probs = config.class_distribution
            print(f"Using class distribution from config: {count_probs}")
    else:
        # Use distribution from config
        count_probs = config.class_distribution
        print(f"Using class distribution from config: {count_probs}")
        
    # Save to config file if specified
    if args.config and args.count_probs:
        config.save_to_file(config_path)
        print(f"Updated configuration saved to {config_path}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect image paths
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    # Use Path.rglob to collect image files
    for ext in image_extensions:
        image_files.extend([str(p) for p in input_dir.rglob(f"*{ext}")])
        image_files.extend([str(p) for p in input_dir.rglob(f"*{ext.upper()}")])  # Also match uppercase extensions
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Count distribution for verification
    count_distribution = {i: 0 for i in range(len(count_probs))}  # Match config distribution classes
    
    # Parse background color
    try:
        bg_color = tuple(int(c) for c in args.bg_color.split(','))
        if len(bg_color) != 3 or not all(0 <= c <= 255 for c in bg_color):
            print(f"Warning: Invalid bg_color format, using default color")
            bg_color = (245, 245, 245)
    except ValueError:
        print(f"Warning: Invalid bg_color format, using default color")
        bg_color = (245, 245, 245)

    # Create collages
    for i in tqdm(range(args.num_collages), desc="Creating collages"):
        canvas_size = (args.canvas_width, args.canvas_height)
        
        # Select receipt count based on probability distribution from config
        receipt_count = random.choices(list(range(len(count_probs))), weights=count_probs)[0]
        
        collage, actual_count = create_receipt_collage(
            image_files, canvas_size, receipt_count,
            bg_color=bg_color, realistic=args.realistic
        )
        
        # Track distribution
        count_distribution[actual_count] += 1
        
        # Save the collage
        output_path = output_dir / f"collage_{i:03d}_{actual_count}_receipts.jpg"
        collage.save(output_path, "JPEG", quality=95)
    
    # Report final receipt count distribution
    print(f"\nActual receipt count distribution:")
    for count, freq in sorted(count_distribution.items()):
        percentage = freq / args.num_collages * 100
        print(f"  {count} receipts: {freq} collages ({percentage:.1f}%)")
    
    print(f"\nCreated {args.num_collages} collages in {output_dir}")

if __name__ == "__main__":
    main()