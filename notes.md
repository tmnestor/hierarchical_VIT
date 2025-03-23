To reduce validation set false negatives (images with receipts being classified as having none), try these
  approaches:

  1. Adjust class weights to penalize false negatives more heavily:
  --level1_weights 1.0,2.5
  2. Use focal loss instead of cross-entropy to focus more on misclassified examples
  3. Data augmentation specifically for minority class samples
  4. Lower the classification threshold (post-processing) from 0.5 to 0.3-0.4
  5. Implement a custom loss function that penalizes false negatives more heavily than false positives
  6. Apply mixup or cutmix augmentation techniques to improve model generalization
  7. Examine the 7 misclassified images - they may share visual characteristics that could help identify the
  problem
  8. Use higher backbone learning rate multiplier (try 0.2 or 0.3) to allow better feature extraction

  The most effective immediate solution is likely adjusting the decision threshold during inference rather than
  retraining.

> Dont implement, just explain;
  Use focal loss instead of cross-entropy to focus more on misclassified examples

⏺ Focal Loss addresses class imbalance and hard examples by down-weighting easy-to-classify examples.

  Unlike standard cross-entropy which treats all examples equally, Focal Loss introduces a modulating factor
  (1-pt)^γ where pt is the model's confidence in the correct class and γ is a focusing parameter (typically
  0.5-5):

  FL(pt) = -α(1-pt)^γ log(pt)

  Key benefits for reducing false negatives:
  - Easy examples (high confidence predictions) contribute less to the loss
  - Hard examples (misclassified or low confidence) receive more attention during training
  - The γ parameter controls how much to focus on hard examples
  - The α parameter can be used as a class weighting factor

  With Focal Loss, the model would focus more on those 7 receipt images it's misclassifying in the validation
  set, potentially learning better discriminative features for distinguishing receipts even when they're
  challenging to detect.

  ⏺ That's a very important insight! Let me summarize the patterns you discovered:

  1. Level 1 False Negatives: Images that have only a single receipt are being misclassified as having no
  receipts (0 when they should be 1+)
  2. Level 2 False Negatives: Images that have exactly 2 receipts are being misclassified as having only 1
  receipt (0 when they should be 2+)

  This suggests that the model struggles with the "boundary cases" - the minimum number required to qualify for
  the positive class in each level. These patterns indicate:

  1. For Level 1: Single receipts might be more challenging to detect than multiple receipts, possibly due to:
    - Less total receipt area in the image
    - Less distinctive visual patterns compared to multiple receipts
  2. For Level 2: The distinction between 1 and 2 receipts is harder than between 1 and 3+ receipts, possibly
  because:
    - With just 2 receipts, there might be less visual evidence of multiplicity
    - The model may need stronger cues for "multiple-ness" than just 2 provides

  Potential solutions:
  - Increase class weights for false negative cases even more
  - Add data augmentation specifically for single receipt and double receipt images
  - Consider a separate binary classifier specifically for distinguishing 1 vs 2 receipts
  - Add attention visualization to see what features the model focuses on

  This discovery will help significantly in improving the model's performance at these critical boundary points.