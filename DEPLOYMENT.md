# 🚀 SecureLens Deployment Guide

## Render.com Deployment

### 1. Ön Hazırlık
- ✅ MongoDB Atlas hesabı ve cluster'ı aktif
- ✅ GitHub repository'sinde kod hazır
- ✅ Render.com hesabı oluşturulmuş

### 2. MongoDB Atlas Ayarları
1. [MongoDB Atlas](https://cloud.mongodb.com) → Login
2. Cluster → Connect → Drivers → Connection String kopyala
3. Username/password ile string'i güncelle:
   ```
   mongodb+srv://<username>:<password>@cluster0.u7deqbd.mongodb.net/securelens?retryWrites=true&w=majority
   ```

### 3. Render.com Deployment
1. [Render.com](https://render.com) → New → Web Service
2. GitHub repository'nizi connect edin
3. Settings:
   - **Name**: `securelens`
   - **Branch**: `master`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app:app --workers 2 --timeout 120`

### 4. Environment Variables
Render Dashboard → Environment → Add:
```bash
# Critical: Set MONGO_URI as SECRET (Hide values option checked)
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/securelens?retryWrites=true&w=majority

# Standard environment variables
DEBUG=false
FLASK_ENV=production
PORT=10000
PYTHONUNBUFFERED=1
```

⚠️ **IMPORTANT**: MONGO_URI'yi mutlaka **Secret** olarak işaretleyin!

### 5. Deploy!
- Save changes → Otomatik deploy başlar
- Logs'u takip edin
- 3-5 dakika sonra live URL alırsınız

### 6. Test Endpoints
Deployment sonrası test edin:
```bash
# Health check
GET https://your-app.onrender.com/health

# AI Status
GET https://your-app.onrender.com/ai-status

# Homepage
GET https://your-app.onrender.com/
```

### 7. Demo URL'ler
Deployment sonrası LinkedIn paylaşımı için:
- **Live Demo**: `https://your-app-name.onrender.com`
- **API Health**: `https://your-app-name.onrender.com/health`
- **Dashboard**: `https://your-app-name.onrender.com/dashboard`

> **Not**: `your-app-name` kısmını Render'da seçtiğiniz app ismi ile değiştirin.

## Troubleshooting

### MongoDB Connection Hatası
- Environment variable'ları kontrol edin
- MongoDB Atlas'ta IP whitelist'i kontrol edin (0.0.0.0/0 ekleyin)
- Connection string format'ını kontrol edin

### Build Hatası
- `requirements.txt` dependencies'ini kontrol edin
- Python version (3.11) uyumluluğunu kontrol edin

### Timeout Hatası
- Gunicorn timeout'u artırın: `--timeout 120`
- Model loading işlemini optimize edin

## LinkedIn Paylaşım Template

```
🚀 SecureLens: AI-Powered Security Analysis Platform'unu paylaşmaktan gurur duyuyorum!

🔗 Live Demo: https://your-app-name.onrender.com
🛡️ URL, Email ve Dosya güvenlik analizi
🤖 Machine Learning destekli hibrit AI
📊 Real-time dashboard ve analytics
⚡ Modern, responsive design

Tech Stack:
- Python, Flask, MongoDB Atlas
- PyTorch, Transformers, scikit-learn
- TailwindCSS, JavaScript
- Deployed on Render.com

#AI #CyberSecurity #MachineLearning #Python #WebDevelopment #Innovation

Proje GitHub: https://github.com/SalihKocak/securelens
```

## Performance Metrics
- ⚡ Build time: ~3-5 minutes
- 🚀 Cold start: ~10-15 seconds
- 📈 Response time: <3 seconds
- 💾 Memory usage: ~500MB
- 🔄 Auto-scaling: Available 