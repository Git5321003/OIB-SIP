# Data Dictionary

## 📋 Column Descriptions

| Column | Data Type | Description |
|--------|-----------|-------------|
| App | String | Name of the application |
| Category | String | Category the app belongs to |
| Rating | Float | Overall user rating (0-5) |
| Reviews | Integer | Number of user reviews |
| Size | Float | Size of the app in megabytes |
| Installs | Integer | Number of installations |
| Type | String | Free or Paid |
| Price | Float | Price in USD (0 for free apps) |
| Content Rating | String | Appropriate audience (Everyone, Teen, etc.) |
| Genres | String | App genres (can be multiple) |
| Last Updated | Date | Date of last update |
| Current Ver | String | Current version number |
| Android Ver | String | Minimum required Android version |

## 🔧 Derived Features
- `Days_Since_Update`: Days since last update
- `Size_Category`: Categorical size groups
- `Install_Category`: Categorical install groups
- `Is_Free`: Boolean for free apps
