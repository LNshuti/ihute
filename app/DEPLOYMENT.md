# Deployment Guide - Hugging Face Spaces

## Updated Dashboard Features

✨ **NEW:** Demographics tab with income distribution, poverty rates, and behavioral impact analysis across 376 Nashville ZCTAs!

## Prerequisites

1. **Hugging Face Account**: Sign up at [huggingface.co](https://huggingface.co)
2. **Git Configured**: Ensure git is installed and configured with your credentials

## Deployment Steps

### Option 1: Deploy via Web Interface (Recommended)

1. **Create a New Space**
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space)
   - Space name: `nashville-incentive-simulation` (or your preferred name)
   - License: MIT
   - SDK: Gradio
   - SDK version: 4.44.0
   - Make it Public or Private
   - Click "Create Space"

2. **Upload Files**

   Upload all files from the `app/` directory:

   **Required files:**
   - `README.md` ✅ (Updated with Demographics tab)
   - `requirements.txt` ✅
   - `app.py` ✅ (Updated with Demographics tab)
   - `database.py` ✅ (Updated with sample demographics)
   - `commuting_zones.py`
   - `nashville_sim_components.py`
   - `nashville_sim_data.py`
   - `nashville_sim_integration.py`

   **Required directory:**
   - `components/` folder (including the NEW `demographics.py`) ✅

3. **Upload via Web**
   - Click "Files" tab in your Space
   - Click "Add file" → "Upload files"
   - Drag and drop all files from `app/` directory
   - Click "Commit changes to main"

4. **Wait for Build**
   - Hugging Face will automatically build and deploy
   - Check the "Logs" tab for build progress
   - Usually takes 2-3 minutes

5. **Verify Deployment**
   - Once built, click the "App" tab
   - Navigate to the "Demographics" tab ✨
   - Verify all visualizations load correctly

### Option 2: Deploy via Git (Advanced)

```bash
# 1. Clone your Hugging Face Space (replace USERNAME and SPACE_NAME)
git clone https://huggingface.co/spaces/USERNAME/SPACE_NAME
cd SPACE_NAME

# 2. Copy app files
cp -r /path/to/ihute/app/* .

# 3. Commit and push
git add .
git commit -m "Add Demographics tab with population-dyna integration"
git push
```

### Option 3: Update Existing Space

If you already have a Space deployed:

```bash
# 1. Clone your existing Space
git clone https://huggingface.co/spaces/LeonceNsh/ihute
cd ihute

# 2. Update files
cp /path/to/ihute/app/app.py .
cp /path/to/ihute/app/database.py .
cp /path/to/ihute/app/README.md .
cp /path/to/ihute/app/components/demographics.py components/
cp /path/to/ihute/app/components/__init__.py components/

# 3. Commit and push
git add .
git commit -m "Update: Add Demographics tab with 376 ZCTA analysis"
git push
```

## Files Updated for Demographics Tab

### New Files
- ✅ `components/demographics.py` - 8 visualization functions for demographics analysis

### Updated Files
- ✅ `app.py` - Added Demographics tab, updated imports
- ✅ `database.py` - Added sample demographics data (376 ZCTAs)
- ✅ `components/__init__.py` - Added demographics imports
- ✅ `README.md` - Added Demographics feature description

## Verification Checklist

After deployment, verify:

- [ ] App loads without errors
- [ ] **Demographics tab** appears (3rd tab)
- [ ] Summary card displays 376 ZCTAs
- [ ] Income distribution chart shows 5 quintiles
- [ ] Poverty distribution chart renders
- [ ] Behavioral impact chart shows VOT + β
- [ ] ZCTA table displays top 20 entries
- [ ] All other tabs still work (Incentive Analytics, Behavioral Calibration, etc.)

## Troubleshooting

### Issue: Demographics tab shows "No data available"

**Solution:** The database.py has been updated with sample demographics data. If you see this error:
1. Verify `database.py` was uploaded with the new `dim_demographics` table creation
2. Check the build logs for any SQL errors
3. Restart the Space from Settings → Factory reboot

### Issue: Import error for demographics module

**Solution:**
1. Ensure `components/demographics.py` was uploaded
2. Ensure `components/__init__.py` includes the demographics imports
3. Check file permissions (should be readable)

### Issue: "Module 'components' has no attribute..."

**Solution:** Clear the build cache:
1. Go to Settings in your Space
2. Click "Factory reboot"
3. Wait for rebuild

## Current Space

**Existing Space:** [huggingface.co/spaces/LeonceNsh/ihute](https://huggingface.co/spaces/LeonceNsh/ihute)

This Space can be updated with the new Demographics tab following Option 3 above.

## Demo Data vs Production Data

**Demo Mode (Hugging Face):**
- Uses sample data generated in `database.py`
- 376 synthetic ZCTAs with realistic distributions
- All features work, but data is simulated

**Production Mode (Local with warehouse.duckdb):**
- Uses real population-dyna demographic data
- 376 actual Tennessee ZCTAs
- Real poverty rates (2020-2022)
- Estimated income from poverty correlations

To use production data on Hugging Face (if database is <100MB):
1. Upload `warehouse.duckdb` to your Space
2. The app will automatically use it instead of sample data

## Post-Deployment

After successful deployment:

1. **Share your Space:**
   - Copy the URL: `https://huggingface.co/spaces/USERNAME/SPACE_NAME`
   - Add to your portfolio
   - Share on social media with #GradioApp tag

2. **Optional Enhancements:**
   - Pin the Space to keep it always running
   - Enable GPU (Settings) if needed for future ML features
   - Add a custom thumbnail image

3. **Monitor Usage:**
   - Check Analytics tab for visitor stats
   - Review build logs for errors
   - Update based on user feedback

## Support

If you encounter issues:

1. Check Hugging Face Spaces docs: [hf.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
2. Review build logs in your Space
3. Test locally first: `cd app && python app.py`

## Success!

Once deployed, your dashboard will feature:
- ✅ Original 6 tabs (Traffic, Incentives, Behavioral, Simulation, Metrics, Map)
- ✨ **NEW Demographics tab** with income/poverty analysis
- 📊 5 interactive visualizations showing behavioral heterogeneity
- 🎯 Production-ready with sample data included

---

**Updated:** 2026-01-19
**Status:** Ready for deployment ✅
