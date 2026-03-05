"""
GeoVision PRO — Flask + OpenCV Shape Detection Server
Ishlatish: python app.py
"""

import cv2
import numpy as np
import base64
import json
import math
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
#  SHAKL MA'LUMOTLAR BAZASI
# ─────────────────────────────────────────────
SHAPE_DB = {
    "Doira":               {"icon": "○", "color": "#00bcd4", "en": "Circle",       "desc": "Barcha nuqtalari markazdan teng masofada.", "formula": "A=πr²  P=2πr"},
    "Ellips":              {"icon": "⊙", "color": "#26c6da", "en": "Ellipse",      "desc": "Ikkita fokusli cho'ziq egri shakl.",         "formula": "A=πab"},
    "Teng tomonli △":     {"icon": "▲", "color": "#ff7043", "en": "Equilateral",   "desc": "3 teng tomon, har burchak 60°.",             "formula": "A=(√3/4)a²"},
    "To'g'ri burchakli △":{"icon": "◺", "color": "#ffa726", "en": "Right △",      "desc": "Bir burchagi 90°. Pifagor: a²+b²=c².",      "formula": "A=ab/2"},
    "Uchburchak":          {"icon": "△", "color": "#ffd600", "en": "Triangle",     "desc": "3 burchak, yig'indisi 180°.",                "formula": "A=bh/2  P=a+b+c"},
    "Kvadrat":             {"icon": "□", "color": "#66bb6a", "en": "Square",       "desc": "4 teng tomon, 4 ta 90° burchak.",            "formula": "A=a²  P=4a"},
    "To'rtburchak":        {"icon": "▭", "color": "#4caf50", "en": "Rectangle",   "desc": "Qarama-qarshi tomonlar teng, 90° burchaklar.","formula": "A=ab  P=2(a+b)"},
    "Romb":                {"icon": "◇", "color": "#81c784", "en": "Rhombus",     "desc": "4 teng tomon, burchaklar 90° emas.",          "formula": "A=d₁d₂/2  P=4a"},
    "Trapeziya":           {"icon": "⏢", "color": "#aed581", "en": "Trapezoid",   "desc": "Bitta juft parallel tomon.",                  "formula": "A=(a+b)h/2"},
    "Pentagon":            {"icon": "⬠", "color": "#ffd600", "en": "Pentagon",    "desc": "5 tomon, ichki burchaklar 540°.",             "formula": "A=(5/4)a²cot(36°)"},
    "Olti burchak":        {"icon": "⬡", "color": "#00e676", "en": "Hexagon",     "desc": "6 teng tomon, asal uyasi shakli.",            "formula": "A=(3√3/2)a²"},
    "Yetti burchak":       {"icon": "◉", "color": "#69f0ae", "en": "Heptagon",    "desc": "7 tomon va 7 burchak.",                       "formula": "P=(n-2)×180°"},
    "Sakkiz burchak":      {"icon": "⯃", "color": "#b2ff59", "en": "Octagon",     "desc": "8 teng tomon.",                               "formula": "A=2(1+√2)a²"},
    "Ko'pburchak":         {"icon": "◈", "color": "#ea80fc", "en": "Polygon",     "desc": "Ko'p tomonli shakl.",                         "formula": "A=na²/(4tan(π/n))"},
}

# ─────────────────────────────────────────────
#  ASOSIY SHAKL ANIQLASH FUNKSIYASI (OpenCV)
# ─────────────────────────────────────────────
def detect_shapes(img_array, sensitivity=80, algo="hybrid"):
    """
    OpenCV yordamida geometrik shakllarni aniqlaydi.
    img_array: numpy BGR array
    """
    results = []
    h, w = img_array.shape[:2]

    # 1. Kulrang rangga o'tkazish
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # 2. Gaussli xira (shovqinni kamaytirish)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

    # 3. Chegarani aniqlash — algoritmga qarab
    if algo == "adaptive":
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 4
        )
    elif algo == "canny":
        edges = cv2.Canny(blurred, 30, 90)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.dilate(binary, kernel, iterations=1)
    else:  # hybrid (default — eng aniq)
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 4
        )
        # + Otsu threshold
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Ikkalasini birlashtirish
        binary = cv2.bitwise_or(adaptive, otsu)

    # 4. Morphological tozalash
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Konturlarni topish (OpenCV eng kuchli joyi)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Minimum maydoni: kamera maydoning 0.3% dan katta bo'lsin
    min_area = w * h * 0.003 * (sensitivity / 80.0)
    max_area = w * h * 0.90

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # Bounding box
        x, y, bw, bh = cv2.boundingRect(cnt)

        # Juda ingichka shakllarni o'tkazib yuborish
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
        if aspect < 0.10:
            continue

        # Konveks hull maydoni
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area < 10:
            continue

        # Asosiy metrikalar
        solidity    = area / hull_area
        perimeter   = cv2.arcLength(cnt, True)
        if perimeter < 1:
            continue
        circularity = (4 * math.pi * area) / (perimeter ** 2)
        rect_fill   = area / (bw * bh) if bw * bh > 0 else 0
        elongation  = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0

        # Solidity tekshirish — noisy shakllarni rad etish
        if solidity < 0.50:
            continue

        # Polygon approxiamtion — eng aniq aniqlash
        # epsilon ni adaptiv hisoblash
        epsilon = 0.025 * perimeter
        approx  = cv2.approxPolyDP(cnt, epsilon, True)
        verts   = len(approx)

        # Burchaklarni hisoblash
        ang_list = []
        for i in range(verts):
            A = approx[(i - 1) % verts][0].astype(float)
            B = approx[i][0].astype(float)
            C = approx[(i + 1) % verts][0].astype(float)
            v1 = A - B
            v2 = C - B
            mag = np.linalg.norm(v1) * np.linalg.norm(v2)
            if mag < 0.001:
                continue
            cos_a = np.clip(np.dot(v1, v2) / mag, -1, 1)
            ang_list.append(math.degrees(math.acos(cos_a)))

        if not ang_list:
            continue

        max_ang = max(ang_list)
        min_ang = min(ang_list)
        has_right = any(abs(a - 90) <= 10 for a in ang_list)
        all_eq60  = verts == 3 and all(abs(a - 60) <= 12 for a in ang_list)

        # Tomonlar uzunligi (normallashtirish)
        side_lens = []
        for i in range(verts):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % verts][0]
            side_lens.append(np.linalg.norm(p1 - p2))

        min_side = min(side_lens) if side_lens else 0
        max_side = max(side_lens) if side_lens else 1
        # Juda kalta tomon bo'lsa — sifatsiz shakl
        if max_side > 0 and min_side / max_side < 0.07:
            continue

        # Tomonlar dispersiyasi
        mean_side = np.mean(side_lens) if side_lens else 1
        side_cv   = np.std(side_lens) / mean_side if mean_side > 0 else 1

        # Ellipse fit (doira/ellips uchun)
        ell_ratio = elongation

        # ── KLASSIFIKATSIYA ──────────────────────────────
        name, conf = classify_shape(
            verts=verts,
            circularity=circularity,
            solidity=solidity,
            rect_fill=rect_fill,
            elongation=elongation,
            max_ang=max_ang,
            min_ang=min_ang,
            has_right=has_right,
            all_eq60=all_eq60,
            ell_ratio=ell_ratio,
            side_cv=side_cv,
        )

        if not name or conf < 62:
            continue

        # Kontur nuqtalarini frontendga yuborish uchun list ga aylantirish
        contour_pts = approx.reshape(-1, 2).tolist()
        hull_pts    = cv2.convexHull(approx).reshape(-1, 2).tolist()

        # Markaz
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + bw // 2, y + bh // 2

        results.append({
            "name":       name,
            "conf":       conf,
            "verts":      verts,
            "center":     [cx, cy],
            "bbox":       {"x": x, "y": y, "w": bw, "h": bh},
            "contour":    contour_pts,
            "hull":       hull_pts,
            "area":       int(area),
            "circularity": round(circularity, 3),
            "solidity":    round(solidity, 3),
            "rect_fill":   round(rect_fill, 3),
            "max_ang":     round(max_ang, 1),
            "info":        SHAPE_DB.get(name, {}),
        })

    # Maydoni bo'yicha saralash (kattadan kichikka)
    results.sort(key=lambda r: r["area"], reverse=True)
    return results[:8]


# ─────────────────────────────────────────────
#  KLASSIFIKATOR
# ─────────────────────────────────────────────
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def classify_shape(verts, circularity, solidity, rect_fill, elongation,
                   max_ang, min_ang, has_right, all_eq60, ell_ratio, side_cv):

    # ── DOIRA / ELLIPS ────────────────────────────────
    if circularity > 0.82 and solidity > 0.88:
        conf = clamp(int(circularity * 108), 72, 99)
        if ell_ratio > 0.80:
            return "Doira", conf
        return "Ellips", clamp(conf - 5, 65, 94)

    if circularity > 0.70 and verts >= 7 and solidity > 0.84:
        return "Doira", clamp(int(circularity * 96), 65, 92)

    if circularity > 0.62 and ell_ratio < 0.72 and verts >= 5 and solidity > 0.80:
        return "Ellips", clamp(int(circularity * 88 + ell_ratio * 8), 60, 88)

    # Yomon solidity → rad
    if solidity < 0.60:
        return None, 0

    # ── UCHBURCHAK ────────────────────────────────────
    if verts == 3:
        if rect_fill < 0.28:
            return None, 0
        if all_eq60:
            return "Teng tomonli △", clamp(74 + int(solidity * 20), 65, 95)
        if has_right and 0.36 < rect_fill < 0.72:
            return "To'g'ri burchakli △", clamp(70 + int(rect_fill * 24), 62, 93)
        if solidity > 0.60:
            return "Uchburchak", clamp(62 + int(solidity * 22), 58, 90)
        return None, 0

    # ── TO'RTBURCHAK ──────────────────────────────────
    if verts == 4:
        if rect_fill < 0.42:
            return None, 0
        if has_right:
            if elongation > 0.82 and rect_fill > 0.78:
                return "Kvadrat", clamp(74 + int(rect_fill * 22), 66, 97)
            if rect_fill > 0.60:
                return "To'rtburchak", clamp(68 + int(rect_fill * 22), 62, 95)
        if solidity > 0.86 and rect_fill < 0.80 and side_cv < 0.22:
            return "Romb", clamp(64 + int(solidity * 20), 58, 90)
        if rect_fill < 0.72 and solidity > 0.72:
            return "Trapeziya", clamp(58 + int(solidity * 18), 52, 84)
        if rect_fill > 0.55:
            return "To'rtburchak", clamp(58 + int(rect_fill * 22), 54, 86)
        return None, 0

    # ── KO'PBURCHAKLAR ────────────────────────────────
    if solidity < 0.68:
        return None, 0

    poly_conf = clamp(64 + int(solidity * 24), 58, 92)
    if verts == 5: return "Pentagon",       poly_conf
    if verts == 6: return "Olti burchak",   poly_conf
    if verts == 7: return "Yetti burchak",  clamp(poly_conf - 4, 54, 88)
    if verts == 8: return "Sakkiz burchak", clamp(poly_conf - 4, 54, 88)

    if verts > 8:
        if circularity > 0.72:
            return "Doira", clamp(int(circularity * 94), 62, 92)
        if solidity > 0.74:
            return "Ko'pburchak", clamp(58 + int(solidity * 18), 52, 82)

    return None, 0


# ─────────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """
    Frontend dan base64 kadr qabul qiladi, shakllarni qaytaradi.
    """
    try:
        data      = request.get_json(force=True)
        img_b64   = data.get("image", "")
        sensitivity = int(data.get("sensitivity", 80))
        algo      = data.get("algo", "hybrid")

        # Base64 → numpy
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]

        img_bytes = base64.b64decode(img_b64)
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Rasm o'qilmadi", "shapes": []})

        shapes = detect_shapes(frame, sensitivity=sensitivity, algo=algo)
        return jsonify({"shapes": shapes, "ok": True})

    except Exception as e:
        return jsonify({"error": str(e), "shapes": []})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "opencv": cv2.__version__})


# ─────────────────────────────────────────────
#  ISHGA TUSHIRISH
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  GeoVision PRO Server — ishga tushmoqda")
    print(f"  OpenCV versiyasi: {cv2.__version__}")
    print("  Brauzerda oching: http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)
