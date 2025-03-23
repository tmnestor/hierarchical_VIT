"""
Global configuration for receipt counting system.
Contains class distribution settings and calibration parameters that can be updated
as the real-world distribution changes, as well as model and training parameters.
"""

import os
import json
import torch

# Default class distribution - can be overridden by environment or config file
# Format: [p0, p1, p2, p3, p4, p5] where p_i is probability of having i receipts
DEFAULT_CLASS_DISTRIBUTION = [0.4, 0.2, 0.2, 0.1, 0.1]

# Default binary distribution - for "multiple receipts or not" classification
# Format: [p0, p1+] where p0 is probability of having 0 receipts, p1+ is probability of having 1+ receipts
DEFAULT_BINARY_DISTRIBUTION = [0.6, 0.4]

# Default model architecture parameters
DEFAULT_MODEL_PARAMS = {
    # Image parameters
    "image_size": 224,
    "normalization_mean": [0.485, 0.456, 0.406],  # ImageNet mean
    "normalization_std": [0.229, 0.224, 0.225],  # ImageNet std
    # Classifier architecture
    "classifier_dims": [768, 512, 256],  # Hidden layer dimensions
    "dropout_rates": [0.4, 0.4, 0.3],  # Dropout rates for each layer
    # Training parameters
    "batch_size": 16,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "num_workers": 4,
    "label_smoothing": 0.1,
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 2,
    "min_lr": 1e-6,
    "gradient_clip_value": 1.0,
    "early_stopping_patience": 5,
    # Reproducibility parameters
    "random_seed": 42,
    "deterministic_mode": True,
}

# Calibration factors will be derived from class distribution automatically


class Config:
    """Global configuration singleton for receipt counting system."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Initialize with defaults directly from this file (single source of truth)
        self.class_distribution = DEFAULT_CLASS_DISTRIBUTION.copy()
        self.binary_mode = False

        # Initialize model params with defaults
        self.model_params = DEFAULT_MODEL_PARAMS.copy()

        # Environment variables can override defaults
        self._load_from_env()

        # Calculate derived values including calibration factors
        self._update_derived_values()

        self._initialized = True

    def _load_from_env(self):
        """Load configuration from environment variables if present."""
        # Format for env vars: RECEIPT_CLASS_DIST="0.3,0.2,0.2,0.1,0.1,0.1"
        if "RECEIPT_CLASS_DIST" in os.environ:
            try:
                dist_str = os.environ["RECEIPT_CLASS_DIST"]
                dist = [float(x) for x in dist_str.split(",")]
                # Validate
                if self.binary_mode:
                    if len(dist) == 2 and abs(sum(dist) - 1.0) < 0.01:
                        self.class_distribution = dist
                else:
                    if len(dist) == 5 and abs(sum(dist) - 1.0) < 0.01:
                        self.class_distribution = dist
            except Exception as e:
                print(f"Warning: Invalid RECEIPT_CLASS_DIST format: {e}")

        # Check for binary mode flag
        if "RECEIPT_BINARY_MODE" in os.environ:
            try:
                self.binary_mode = os.environ["RECEIPT_BINARY_MODE"].lower() in (
                    "true",
                    "1",
                    "yes",
                )
                # If switching to binary mode, update distribution
                if self.binary_mode:
                    self.class_distribution = DEFAULT_BINARY_DISTRIBUTION.copy()
            except Exception as e:
                print(f"Warning: Invalid RECEIPT_BINARY_MODE format: {e}")

        # Load model parameters from environment variables
        env_to_param = {
            "RECEIPT_IMAGE_SIZE": ("image_size", int),
            "RECEIPT_BATCH_SIZE": ("batch_size", int),
            "RECEIPT_LEARNING_RATE": ("learning_rate", float),
            "RECEIPT_NUM_WORKERS": ("num_workers", int),
            "RECEIPT_WEIGHT_DECAY": ("weight_decay", float),
            "RECEIPT_LABEL_SMOOTHING": ("label_smoothing", float),
            "RECEIPT_GRADIENT_CLIP": ("gradient_clip_value", float),
        }
        
        # Process each environment variable
        for env_var, (param_name, convert_func) in env_to_param.items():
            if env_var in os.environ:
                try:
                    self.model_params[param_name] = convert_func(os.environ[env_var])
                except Exception as e:
                    print(f"Warning: Invalid {env_var} format: {e}")

    def load_from_file(self, config_path, silent=False):
        """Load configuration from a JSON file.

        Args:
            config_path: Path to the configuration file
            silent: If True, don't print a message when loading config
        """
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)

            # Check for binary mode flag
            if "binary_mode" in config_data:
                self.binary_mode = bool(config_data["binary_mode"])

            # Update distribution if valid
            if "class_distribution" in config_data:
                dist = config_data["class_distribution"]
                expected_length = 2 if self.binary_mode else 5
                if len(dist) == expected_length and abs(sum(dist) - 1.0) < 0.01:
                    self.class_distribution = dist
                elif not silent:
                    print(
                        f"Warning: Invalid class distribution in config file. Expected {expected_length} values that sum to 1.0"
                    )

            # Load model parameters if present
            if "model_params" in config_data:
                for key, value in config_data["model_params"].items():
                    if key in self.model_params:
                        self.model_params[key] = value
                    elif not silent:
                        print(f"Warning: Unknown model parameter in config file: {key}")

            self._update_derived_values()

            if not silent:
                print(f"Loaded configuration from {config_path}")

        except Exception as e:
            if not silent:
                print(f"Error loading config from {config_path}: {e}")

    def save_to_file(self, config_path):
        """Save current configuration to a JSON file."""
        config_data = {
            "binary_mode": self.binary_mode,
            "class_distribution": self.class_distribution,
            # Include model parameters
            "model_params": self.model_params,
            # We include the derived calibration factors as information only
            "derived_calibration_factors": self.calibration_factors,
        }

        try:
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)
            print(f"Configuration saved to {config_path}")
        except Exception as e:
            print(f"Error saving config to {config_path}: {e}")

    def _update_derived_values(self):
        """Update any derived configuration values."""
        # Calculate inverse weights for CrossEntropyLoss
        self.inverse_weights = [
            1.0 / w if w > 0 else 1.0 for w in self.class_distribution
        ]
        sum_inverse = sum(self.inverse_weights)
        self.normalized_weights = [w / sum_inverse for w in self.inverse_weights]
        self.scaled_weights = [
            w * len(self.class_distribution) for w in self.normalized_weights
        ]  # Scale by num classes

        # Always derive calibration factors from class distribution using a principled approach
        # For calibration factors, we need to counteract the class weighting during inference
        # A principled approach is to use:
        # calibration_factor = prior_probability * sqrt(reference_probability / prior_probability)
        # This balances the influence of the prior while adding compensation for minority classes
        reference_prob = 1.0 / len(self.class_distribution)  # Equal distribution
        self.calibration_factors = [
            prior * ((reference_prob / prior) ** 0.5) if prior > 0 else 1.0
            for prior in self.class_distribution
        ]

        # Normalize calibration factors for better comparison
        max_cal = max(self.calibration_factors)
        self.calibration_factors = [cal / max_cal for cal in self.calibration_factors]

    def get_class_weights_tensor(self, device=None):
        """Get class weights as a tensor for CrossEntropyLoss."""
        weights = torch.tensor(self.scaled_weights)
        if device:
            weights = weights.to(device)
        return weights

    def get_calibration_tensor(self, device=None):
        """Get calibration factors as a tensor for inference."""
        cal = torch.tensor(self.calibration_factors)
        if device:
            cal = cal.to(device)
        return cal

    def get_class_prior_tensor(self, device=None):
        """Get class distribution as a tensor for calibration."""
        prior = torch.tensor(self.class_distribution)
        if device:
            prior = prior.to(device)
        return prior

    def get_model_param(self, param_name, default=None):
        """Get a model parameter by name with optional default value."""
        return self.model_params.get(param_name, default)

    def update_class_distribution(self, new_distribution):
        """Update the class distribution and recalculate derived values."""
        expected_length = 2 if self.binary_mode else 5
        if (
            len(new_distribution) != expected_length
            or abs(sum(new_distribution) - 1.0) > 0.01
        ):
            raise ValueError(
                f"Class distribution must have {expected_length} values that sum to approximately 1.0"
            )

        self.class_distribution = new_distribution
        self._update_derived_values()

    def update_model_param(self, param_name, value):
        """Update a single model parameter."""
        if param_name in self.model_params:
            self.model_params[param_name] = value
            return True
        return False

    def update_model_params(self, params_dict):
        """Update multiple model parameters at once."""
        updated = []
        for key, value in params_dict.items():
            if key in self.model_params:
                self.model_params[key] = value
                updated.append(key)
        return updated

    def set_binary_mode(self, binary=True):
        """Switch between regular classification and binary classification."""
        if binary == self.binary_mode:
            # No change needed
            return

        # Set new mode
        self.binary_mode = binary

        # Reset to appropriate default distribution
        if binary:
            self.class_distribution = DEFAULT_BINARY_DISTRIBUTION.copy()
        else:
            self.class_distribution = DEFAULT_CLASS_DISTRIBUTION.copy()

        # Update derived values
        self._update_derived_values()

        print(
            f"Switched to {'binary' if binary else 'multi-class'} classification mode"
        )

    def explain_calibration(self):
        """Explain how calibration factors are derived from class distribution."""
        reference_prob = 1.0 / len(self.class_distribution)
        explanation = {
            "mode": "binary" if self.binary_mode else "multi-class",
            "class_distribution": self.class_distribution,
            "reference_probability": reference_prob,
            "derivation_steps": [],
        }

        # Calculate the raw factors before normalization
        raw_factors = []
        for i, prior in enumerate(self.class_distribution):
            if prior > 0:
                sqrt_term = (reference_prob / prior) ** 0.5
                raw_factor = prior * sqrt_term
            else:
                raw_factor = 1.0

            raw_factors.append(raw_factor)

            step = {
                "class": i,
                "prior_probability": prior,
                "sqrt_term": sqrt_term if prior > 0 else "N/A",
                "raw_factor": raw_factor,
            }
            explanation["derivation_steps"].append(step)

        # Add normalization step
        max_raw = max(raw_factors)
        explanation["max_raw_factor"] = max_raw
        explanation["normalized_factors"] = self.calibration_factors

        return explanation


# Singleton instance for easy import
config = Config()


def get_config():
    """Get the global configuration instance."""
    return config
