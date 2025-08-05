# Git LFS Guide for Model Weights

## Setting Up Git LFS for Model Weights

### On the Machine WITH Weights (Uploading)

1. **Install Git LFS**
```bash
# Ubuntu/Debian
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# Initialize Git LFS in your user account
git lfs install
```

2. **Track Model Files**
```bash
# In the repository directory
cd jax-for-gpt-oss-20b

# Track safetensor files
git lfs track "*.safetensors"
git lfs track "models/**/*.safetensors"
git lfs track "*.pkl"
git lfs track "jax_params/**"

# This creates/updates .gitattributes
git add .gitattributes
```

3. **Add and Upload Weights**
```bash
# Copy weights to repo (if not already there)
mkdir -p models/gpt-oss-20b
cp -r /root/models/gpt-oss-20b/* models/gpt-oss-20b/

# Add to Git LFS
git add models/
git commit -m "Add GPT-OSS-20B weights via Git LFS"

# Push (this will upload the large files)
git push origin main
# Note: This will take time for 13.5GB of weights
```

### On Another Machine (Downloading)

1. **Install Git LFS**
```bash
# Same installation as above for your OS
git lfs install
```

2. **Clone WITH Weights**
```bash
# Method 1: Clone with automatic LFS download
git clone https://github.com/atsentia/jax-for-gpt-oss-20b.git
cd jax-for-gpt-oss-20b

# The weights should download automatically during clone
# Check if weights are present
ls -lh models/gpt-oss-20b/
```

3. **If Weights Didn't Download Automatically**
```bash
# Method 2: Clone without LFS, then pull
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/atsentia/jax-for-gpt-oss-20b.git
cd jax-for-gpt-oss-20b

# Now pull just the weights you need
git lfs pull --include="models/gpt-oss-20b/*.safetensors"

# Or pull everything
git lfs pull
```

4. **Selective Download (Save Bandwidth)**
```bash
# Clone without downloading LFS files
git clone https://github.com/atsentia/jax-for-gpt-oss-20b.git
cd jax-for-gpt-oss-20b

# Download only specific files
git lfs pull --include="models/gpt-oss-20b/model-00000-of-00002.safetensors"
git lfs pull --include="models/gpt-oss-20b/config.json"

# Download by pattern
git lfs pull --include="*.json"  # Just config files
git lfs pull --include="models/gpt-oss-20b/model-*.safetensors"  # All model files
```

## Checking LFS Status

```bash
# See which files are tracked by LFS
git lfs ls-files

# Check tracking patterns
cat .gitattributes

# See LFS file sizes
git lfs ls-files -s

# Check if files are pointers or actual content
ls -la models/gpt-oss-20b/
# Pointer files will be ~130 bytes
# Actual files will be GB in size
```

## Troubleshooting

### If Clone is Slow
```bash
# Use shallow clone first
git clone --depth 1 https://github.com/atsentia/jax-for-gpt-oss-20b.git
cd jax-for-gpt-oss-20b
git lfs pull
```

### If You Run Out of GitHub LFS Storage
```bash
# Check your LFS storage usage
git lfs env

# Alternative: Use external storage
# 1. Upload weights to HuggingFace Hub or Google Drive
# 2. Use download scripts instead of Git LFS
```

### Converting Existing Repo to Use LFS
```bash
# If you already committed large files without LFS
git lfs migrate import --include="*.safetensors" --everything
git push --force-with-lease origin main
```

## Alternative: Using Download Script

If Git LFS is not available or practical:

```bash
# On the target machine, just use the download script
cd jax-for-gpt-oss-20b
bash scripts/download_model.sh

# This downloads directly from HuggingFace, bypassing Git LFS
```

## For Private Repositories

For private repos, you may need to authenticate:

```bash
# Set up credentials
git config --global credential.helper store

# Or use SSH
git clone git@github.com:your-org/jax-for-gpt-oss-20b-private.git

# For GitHub, you might need a token with LFS access
git clone https://<token>@github.com/your-org/repo.git
```

## Summary Commands

**Quick setup on new machine:**
```bash
# Install Git LFS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

# Clone with weights
git clone https://github.com/atsentia/jax-for-gpt-oss-20b.git
cd jax-for-gpt-oss-20b

# Verify weights downloaded
du -sh models/gpt-oss-20b/
# Should show ~13.5GB

# If not downloaded, pull them
git lfs pull
```

**Note:** GitHub provides 1GB of free LFS storage and 1GB bandwidth per month. For the 13.5GB model, you would need to purchase additional LFS storage or use alternative hosting like HuggingFace Hub.