from models.model_manager import ModelManager

model_manager = ModelManager()
models_df = model_manager.list_models()
print(models_df)
