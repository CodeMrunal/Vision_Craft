# 🚀 Deployment Guide - VisionCraft on GitHub

This guide will help you deploy your VisionCraft image classification app to GitHub and run it on Streamlit Cloud (free hosting).

---

## 📋 Prerequisites

| **Requirement** | **Status** |
|-----------------|------------|
| GitHub Account | Required (free at github.com) |
| Git Installed | Check with `git --version` |
| Project Files Ready | All files in VisionCraft folder |

---

## 🔧 Step 1: Initialize Git Repository

### 1.1 Open Terminal/Command Prompt
Navigate to your project folder:
```bash
cd C:\Users\falgu\OneDrive\Desktop\VisionCraft
```

### 1.2 Initialize Git
```bash
git init
```

### 1.3 Create .gitignore File
Create a `.gitignore` file to exclude unnecessary files:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Model files (too large for GitHub)
*.h5
*.hdf5
*.pb
*.pkl
*.pt
*.pth

# Jupyter Notebook
.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/

# PDF (optional - if you don't want to commit)
# *.pdf
```

---

## 📤 Step 2: Push to GitHub

### 2.1 Create GitHub Repository

1. Go to [github.com](https://github.com) and sign in
2. Click the **"+"** icon in the top right → **"New repository"**
3. Repository name: `VisionCraft` (or your preferred name)
4. Description: "Image Classification Web App using Streamlit and TensorFlow"
5. Choose **Public** (required for free Streamlit Cloud)
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click **"Create repository"**

### 2.2 Add Files and Commit

```bash
# Add all files
git add .

# Commit with message
git commit -m "Initial commit: VisionCraft image classification app"

# Add remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/VisionCraft.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Note:** You'll be prompted for your GitHub username and password (use a Personal Access Token, not your password).

---

## ☁️ Step 3: Deploy on Streamlit Cloud (FREE)

### 3.1 Go to Streamlit Cloud

1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign in with GitHub"**
3. Authorize Streamlit Cloud to access your GitHub account

### 3.2 Deploy Your App

1. Click **"New app"**
2. Select your repository: `YOUR_USERNAME/VisionCraft`
3. Branch: `main`
4. Main file path: `app.py`
5. Click **"Deploy!"**

### 3.3 Wait for Deployment

- Streamlit Cloud will automatically:
  - Install dependencies from `requirements.txt`
  - Build and deploy your app
  - Provide a public URL (e.g., `https://your-app.streamlit.app`)

---

## 📝 Step 4: Add Model File (Important!)

**⚠️ Important:** Since `model.h5` is in `.gitignore` (too large for GitHub), you have two options:

### Option A: Use Streamlit Secrets (Recommended for small models < 100MB)

1. In Streamlit Cloud, go to your app settings
2. Click **"Secrets"**
3. Add your model file using GitHub Releases or upload it separately

### Option B: Use GitHub Releases (For larger models)

1. Go to your GitHub repository
2. Click **"Releases"** → **"Create a new release"**
3. Upload `model.h5` as an asset
4. Update `app.py` to download the model on first run:

```python
import urllib.request
import os

MODEL_URL = "https://github.com/YOUR_USERNAME/VisionCraft/releases/download/v1.0/model.h5"
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.success("Model downloaded!")
```

### Option C: Use External Storage (Google Drive, Dropbox, etc.)

Upload `model.h5` to cloud storage and update the code to download it.

---

## 🔄 Step 5: Update Code for Cloud Deployment

### 5.1 Update app.py for Model Loading

Add this at the beginning of your `main()` function:

```python
def main() -> None:
    st.set_page_config(page_title="VisionCraft - Image Classifier", layout="wide")
    
    # Check for model in multiple locations
    model_path = "model.h5"
    if not os.path.exists(model_path):
        # Try alternative paths or download
        st.error("Model file not found. Please ensure model.h5 is available.")
        st.stop()
    # ... rest of your code
```

### 5.2 Ensure requirements.txt is Complete

Make sure `requirements.txt` includes all dependencies:

```
streamlit>=1.36.0
tensorflow>=2.12.0
numpy>=1.24.0
Pillow>=9.5.0
pandas>=2.0.0
altair>=5.0.0
```

---

## 🌐 Step 6: Access Your Live App

Once deployed, your app will be available at:
```
https://YOUR_USERNAME-VisionCraft-main-app-XXXXXX.streamlit.app
```

You can:
- Share this URL with anyone
- Use it in your portfolio
- Access it from any device

---

## 🔧 Troubleshooting

| **Issue** | **Solution** |
|-----------|--------------|
| **Deployment fails** | Check `requirements.txt` has all dependencies |
| **Model not found** | Ensure model.h5 is accessible (see Step 4) |
| **App crashes** | Check Streamlit Cloud logs in the app dashboard |
| **Slow loading** | Model file might be too large; consider using smaller model or external storage |
| **Import errors** | Verify all packages in `requirements.txt` |

---

## 📊 Alternative Deployment Options

### Option 1: Heroku
- More complex setup
- Requires `Procfile` and `runtime.txt`
- Free tier available (with limitations)

### Option 2: AWS/Azure/GCP
- More control and scalability
- Requires cloud account setup
- May incur costs

### Option 3: Docker + Any Cloud
- Containerized deployment
- Works on any platform
- Requires Docker knowledge

**Recommendation:** Streamlit Cloud is the easiest and free option for Streamlit apps!

---

## 🔐 Security Notes

1. **Don't commit sensitive data:**
   - API keys
   - Personal information
   - Large model files (use `.gitignore`)

2. **Use Streamlit Secrets:**
   - For configuration
   - For API keys
   - For sensitive paths

3. **Model Privacy:**
   - If your model is proprietary, consider private repository
   - Or use private cloud storage

---

## 📚 Quick Reference Commands

```bash
# Check git status
git status

# Add files
git add .

# Commit changes
git commit -m "Your commit message"

# Push to GitHub
git push origin main

# Pull latest changes
git pull origin main

# View deployment logs (in Streamlit Cloud dashboard)
# Go to: share.streamlit.io → Your App → Logs
```

---

## ✅ Deployment Checklist

- [ ] Git repository initialized
- [ ] `.gitignore` file created
- [ ] All files committed
- [ ] Repository pushed to GitHub
- [ ] Streamlit Cloud account connected
- [ ] App deployed on Streamlit Cloud
- [ ] Model file accessible (via secrets/releases/storage)
- [ ] App tested and working
- [ ] Public URL shared/added to portfolio

---

## 🎉 You're Done!

Your VisionCraft app is now live on the internet! 🚀

**Next Steps:**
- Share your app URL
- Add it to your portfolio/resume
- Continue improving the app
- Deploy updates by pushing to GitHub (auto-deploys)

---

**👩‍💻 Made by Falguni Shinde**





