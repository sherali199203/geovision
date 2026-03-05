# GeoVision PRO — Python + OpenCV Server

## Fayl tuzilmasi
```
geovision_server/
├── app.py              ← Asosiy server (Flask + OpenCV)
├── requirements.txt    ← Python kutubxonalar
├── Procfile            ← Server deploy konfiguratsiyasi
└── templates/
    └── index.html      ← Brauzer interfeysi
```

---

## Kompyuterda ishga tushirish (test uchun)

```bash
# 1. Python kutubxonalarini o'rnatish
pip install -r requirements.txt

# 2. Serverni ishga tushirish
python app.py

# 3. Brauzerda ochish
# http://localhost:5000
```

---

## Railway.app ga deploy qilish (BEPUL, eng oson)

1. **https://railway.app** ga kiring → GitHub bilan login
2. **New Project → Deploy from GitHub repo**
3. Bu papkani GitHub ga yuklang (yoki zip drag-and-drop)
4. Railway avtomatik `requirements.txt` ni o'qiydi va o'rnatadi
5. **Deploy** tugmasini bosing
6. 2-3 daqiqada tayyor URL beriladi

---

## Render.com ga deploy qilish (BEPUL)

1. **https://render.com** ga kiring
2. **New → Web Service**
3. GitHub repo ni ulang
4. Sozlamalar:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
5. **Create Web Service** → 3-5 daqiqada tayyor

---

## VPS ga deploy qilish (Ubuntu)

```bash
# Server ga SSH orqali kiring
ssh user@your-server-ip

# Papkani nusxa oling
git clone https://github.com/sizning-repo/geovision_server
cd geovision_server

# O'rnatish
pip install -r requirements.txt

# Ishga tushirish (doim ishlashi uchun)
gunicorn app:app --bind 0.0.0.0:5000 --workers 2 --daemon

# Nginx bilan (ixtiyoriy)
# /etc/nginx/sites-available/geovision ga qo'shing:
# proxy_pass http://127.0.0.1:5000;
```

---

## API

| Endpoint | Method | Tavsif |
|---|---|---|
| `/` | GET | Asosiy interfeys |
| `/api/detect` | POST | Shakl aniqlash (base64 rasm qabul qiladi) |
| `/api/health` | GET | Server holati va OpenCV versiyasi |
