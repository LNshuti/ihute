#!/bin/bash
# Deployment script for Hugging Face Spaces
# Updates existing Space with Demographics tab

set -e  # Exit on error

echo "🚀 Hugging Face Spaces Deployment Script"
echo "=========================================="
echo ""

# Check if HF Space URL is provided
if [ -z "$1" ]; then
    echo "❌ Error: Please provide your Hugging Face Space URL"
    echo ""
    echo "Usage:"
    echo "  ./deploy_to_hf.sh https://huggingface.co/spaces/USERNAME/SPACE_NAME"
    echo ""
    echo "Example:"
    echo "  ./deploy_to_hf.sh https://huggingface.co/spaces/LeonceNsh/ihute"
    exit 1
fi

HF_SPACE_URL="$1"
TEMP_DIR="/tmp/hf_deploy_$$"

echo "📦 Target Space: $HF_SPACE_URL"
echo ""

# Clone the Space
echo "📥 Cloning Space..."
git clone "$HF_SPACE_URL" "$TEMP_DIR"
cd "$TEMP_DIR"

echo "✅ Cloned successfully"
echo ""

# Get the app directory path
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/app"

# Copy updated files
echo "📄 Copying updated files..."
echo "  - app.py (with Demographics tab)"
cp "$APP_DIR/app.py" .

echo "  - database.py (with sample demographics)"
cp "$APP_DIR/database.py" .

echo "  - README.md (updated features)"
cp "$APP_DIR/README.md" .

echo "  - components/__init__.py"
cp "$APP_DIR/components/__init__.py" components/

echo "  - components/demographics.py (NEW)"
cp "$APP_DIR/components/demographics.py" components/

echo "✅ Files copied"
echo ""

# Show changes
echo "📊 Changes to be committed:"
git status --short

echo ""
read -p "👉 Proceed with deployment? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Deployment cancelled"
    rm -rf "$TEMP_DIR"
    exit 0
fi

# Commit and push
echo "📤 Committing changes..."
git add .
git commit -m "✨ Add Demographics tab with population-dyna integration

- NEW: Demographics analysis tab (376 Nashville ZCTAs)
- Income distribution by quintile
- Poverty rate visualization
- Behavioral impact analysis (VOT + incentive sensitivity)
- Sample demographics data included
- Updated dashboard with 7 tabs total"

echo "🚀 Pushing to Hugging Face..."
git push

echo ""
echo "✅ Deployment successful!"
echo ""
echo "🌐 Your Space is being rebuilt at: $HF_SPACE_URL"
echo "⏱️  Build usually takes 2-3 minutes"
echo ""
echo "📋 Next steps:"
echo "  1. Visit $HF_SPACE_URL"
echo "  2. Check 'Logs' tab for build progress"
echo "  3. Once ready, navigate to 'Demographics' tab ✨"
echo ""

# Cleanup
rm -rf "$TEMP_DIR"

echo "🎉 Done!"
