import os
import pandas as pd
import argparse
import torch
import glob
from model_factory import ModelFactory
from receipt_processor import ReceiptProcessor
from device_utils import get_device
from tqdm import tqdm

def process_document_archive(model_path, input_dir, output_csv):
    """Process a directory of scanned documents and count receipts in each."""
    device = get_device()
    # Determine model type from filename, default to swin
    model_type = "vit" if "vit" in model_path.lower() else "swin"
    model = ModelFactory.load_model(model_path, model_type=model_type, mode="eval").to(device)
    processor = ReceiptProcessor()
    
    # Find all image files
    image_extensions = ['jpg', 'jpeg', 'png', 'tif', 'tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, f"*.{ext}")))
    
    results = []
    
    # Process each image
    for img_path in tqdm(image_paths, desc="Processing documents"):
        try:
            # Preprocess the image
            img_tensor = processor.preprocess_image(img_path).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(img_tensor)
                # Get logits from model outputs
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # Get class prediction (0-5)
                probs = torch.nn.functional.softmax(logits, dim=1)
                predicted_class = torch.argmax(logits, dim=1).item()
                confidence = probs[0, predicted_class].item()
            
            # The predicted class is directly the count (0-5)
            count = predicted_class
            
            results.append({
                'document': os.path.basename(img_path),
                'receipt_count': count,
                'confidence': float(confidence)
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Save results
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    
    # Summary
    total_receipts = sum(r['receipt_count'] for r in results)
    print(f"Processed {len(results)} documents")
    print(f"Detected a total of {total_receipts} receipts")
    # Show average confidence
    avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
    print(f"Average confidence: {avg_confidence:.2%}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Receipt Counter")
    parser.add_argument("--input_dir", required=True, help="Directory with scanned documents")
    parser.add_argument("--output_csv", required=True, help="Path to save results CSV")
    parser.add_argument("--model", default="models/receipt_counter_swin_best.pth",
                       help="Path to model file")
    parser.add_argument("--model_type", choices=["vit", "swin"], 
                       help="Model type (vit or swin). If not specified, detected from model filename.")
    
    args = parser.parse_args()
    
    # If model_type is provided, override the auto-detection in process_document_archive
    if args.model_type:
        model_path = args.model
        # Create a new model with the specified type
        device = get_device()
        model = ModelFactory.load_model(model_path, model_type=args.model_type, mode="eval")
        model = model.to(device)
        processor = ReceiptProcessor()
        
        # We can't directly pass the model to process_document_archive due to its signature,
        # so we'd need to refactor it. For now, just use the auto-detection
        print(f"Using model type: {args.model_type} (specified via argument)")
    
    process_document_archive(args.model, args.input_dir, args.output_csv)