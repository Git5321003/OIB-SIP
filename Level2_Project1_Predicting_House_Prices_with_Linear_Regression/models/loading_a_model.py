from models.model_manager import ModelManager

# Initialize model manager
model_manager = ModelManager()

# Load latest model
model_data = model_manager.load_model("housing_price_predictor", "latest")

# Or load specific version
model_data = model_manager.load_model("housing_price_predictor", "v1.0.0")
