# Configuration Files

This folder contains all configuration files for the HeartGuard AI project.

## 📁 Files

### `config.env`
Main configuration file for the machine learning pipeline and data processing.

**Sections:**
- **Data Configuration**: Dataset URLs, paths, and directories
- **Model Configuration**: Test size, random state, cross-validation settings
- **SMOTE Configuration**: Class balancing parameters
- **Feature Selection**: Feature selection parameters
- **Logging Configuration**: Log levels and file paths
- **Visualization Configuration**: Plot settings and styles
- **Model Parameters**: ML model hyperparameters
- **Performance Thresholds**: Model performance requirements
- **Training Options**: Training pipeline settings

### `flask_config.env`
Flask web application specific configuration.

**Sections:**
- **Flask Settings**: Environment, debug mode, secret key
- **Server Configuration**: Host, port, debug settings
- **Model Configuration**: Model file paths and directories
- **Data Configuration**: Dataset URLs and paths
- **Logging Configuration**: Flask-specific logging
- **API Configuration**: API timeout and content limits
- **CORS Configuration**: Cross-origin resource sharing
- **Cache Configuration**: Caching settings

## 🔧 Usage

### Loading Configuration in Python
```python
from dotenv import load_dotenv
import os

# Load main config
load_dotenv('configs/config.env')

# Load Flask config
load_dotenv('configs/flask_config.env')

# Access variables
data_url = os.getenv('DATA_URL')
model_dir = os.getenv('MODEL_DIR')
```

### Using the Config Class
```python
from app.config import config

# Access configuration
print(config.data.DATA_URL)
print(config.model.TEST_SIZE)
print(config.logging.LOG_LEVEL)
```

## 📝 Configuration Management

### Environment Variables
All configuration values are loaded as environment variables using `python-dotenv`.

### Default Values
Each configuration has sensible default values if not specified in the `.env` files.

### Type Conversion
The config class automatically converts string values to appropriate types:
- `int()` for integer values
- `float()` for decimal values
- `bool()` for boolean values
- `eval()` for complex types like tuples

## 🚀 Deployment

### Development
Use the default values in the `.env` files for development.

### Production
Override environment variables for production:
```bash
export FLASK_ENV=production
export DEBUG=false
export LOG_LEVEL=WARNING
```

### Docker
Set environment variables in Docker:
```dockerfile
ENV FLASK_ENV=production
ENV DEBUG=false
ENV MODEL_DIR=/app/models
```

## 🔒 Security

### Secret Keys
- Flask secret key is set in `flask_config.env`
- Change the secret key for production deployments
- Never commit secret keys to version control

### Sensitive Data
- Database credentials (if added)
- API keys (if added)
- Production URLs and paths

## 📊 Configuration Hierarchy

1. **Environment Variables** (highest priority)
2. **Config Files** (`.env` files)
3. **Default Values** (hardcoded in config classes)

## 🛠️ Customization

### Adding New Configuration
1. Add the variable to the appropriate `.env` file
2. Add the variable to the corresponding config class
3. Use `os.getenv()` with a default value
4. Update documentation

### Modifying Existing Configuration
1. Edit the `.env` file
2. Restart the application
3. Test the changes

## 📋 Best Practices

- ✅ Use descriptive variable names
- ✅ Group related configurations together
- ✅ Provide sensible default values
- ✅ Document configuration options
- ✅ Use environment-specific configs
- ❌ Don't hardcode sensitive information
- ❌ Don't commit production secrets
- ❌ Don't use complex types without proper conversion
