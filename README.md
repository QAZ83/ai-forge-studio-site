# 🚀 AI Forge Studio - Advanced UI Dashboard

<div align="center">

![AI Forge Studio](https://img.shields.io/badge/AI%20Forge%20Studio-v1.0.0-00d9ff?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-00ffcc?style=for-the-badge)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

**منصة احترافية متقدمة لتشغيل نماذج الذكاء الاصطناعي على RTX 5090 مع دعم CUDA و TensorRT و Vulkan SDK**

[المعاينة المباشرة](https://aiforgestudio.net) | [التوثيق](#features) | [دليل C++/Qt](CPP-Qt-Integration-Guide.md)

</div>

---

## ✨ المميزات

### 🎨 تصميم عصري واحترافي
- **Dark Theme** - تصميم داكن بألوان Cyan و Teal المميزة
- **Futuristic UI** - واجهة مستخدم مستقبلية مع تأثيرات Circuit Board
- **Responsive Design** - متجاوب تماماً مع جميع الشاشات
- **Smooth Animations** - حركات وانتقالات سلسة ومميزة

### 📊 لوحات تحكم تفاعلية
- **GPU Accelerated Output** - رسومات ثلاثية الأبعاد لتمثيل نشاط GPU
- **System Status** - مؤشرات دائرية لاستخدام GPU و CPU
- **Training Charts** - رسوم بيانية لتتبع عملية التدريب
- **Real-time Monitoring** - مراقبة في الوقت الفعلي للنظام

### 🔧 أدوات تطوير متقدمة
- **CUDA Integration** - دعم كامل لـ CUDA 12.4 مع أمثلة برمجية
- **TensorRT Engine** - تحويل وتحسين النماذج للحصول على أعلى أداء
- **Vulkan SDK** - تسريع الرسومات والحسابات باستخدام Vulkan
- **Model Inference** - واجهة تفاعلية لتشغيل النماذج المختلفة
- **C++/Qt Application** - تطبيق سطح مكتب متكامل

### ⚡ أداء عالي
- **Optimized Canvas** - رسومات Canvas محسّنة
- **Chart.js Integration** - رسوم بيانية احترافية
- **Lazy Loading** - تحميل ذكي للموارد
- **60 FPS Animations** - حركات بمعدل 60 إطار في الثانية

---

## 🚀 البدء السريع

### المتطلبات
- متصفح حديث (Chrome, Firefox, Edge, Safari)
- لا حاجة لأي خادم - يعمل مباشرة!

### التثبيت

```bash
# استنساخ المشروع
git clone https://github.com/yourusername/ai-forge-studio-site.git

# الانتقال إلى المجلد
cd ai-forge-studio-site

# فتح في المتصفح
open index.html
# أو
start index.html
```

### البنية الأساسية

```
ai-forge-studio-site/
├── index.html                      # الصفحة الرئيسية
├── dashboard.html                  # لوحة التحكم الكاملة
├── cuda-integration.html           # صفحة CUDA
├── tensorrt.html                   # صفحة TensorRT
├── vulkan.html                     # صفحة Vulkan SDK
├── inference.html                  # صفحة Model Inference
├── CPP-Qt-Integration-Guide.md     # دليل C++/Qt الشامل
├── css/
│   ├── style.css                   # التنسيقات الرئيسية
│   └── dashboard.css               # تنسيقات Dashboard
├── js/
│   ├── app.js                      # الوظائف الرئيسية
│   ├── interactive.js              # النظام التفاعلي
│   ├── cuda-integration.js         # وظائف CUDA
│   ├── tensorrt.js                 # وظائف TensorRT
│   └── inference.js                # محرك الاستدلال
└── README.md                       # هذا الملف
```

---

## 🎯 الاستخدام

### الصفحة الرئيسية
```html
<!-- فتح الصفحة الرئيسية -->
<a href="index.html">AI Forge Studio - Main</a>
```

### لوحة التحكم الكاملة
```html
<!-- فتح Dashboard -->
<a href="dashboard.html">AI Forge Studio - Dashboard</a>
```

### التخصيص

#### تغيير الألوان
```css
/* في ملف css/style.css */
:root {
    --primary-cyan: #00d9ff;    /* اللون الأساسي */
    --primary-teal: #00ffcc;    /* اللون الثانوي */
    --accent-blue: #0099ff;     /* لون التمييز */
}
```

#### إضافة مؤشر جديد
```javascript
// في ملف js/app.js
class CustomMetric {
    constructor() {
        this.value = 0;
        this.update();
    }

    update() {
        // منطقك هنا
    }
}
```

---

## 📦 المكونات

### 1. GPU Visualization
يعرض شبكة ثلاثية الأبعاد متحركة تمثل نشاط GPU في الوقت الفعلي.

```javascript
// استخدام مكون GPU Visualization
const gpuViz = new GPUVisualization('gpuCanvas');
```

### 2. Circular Progress
مؤشرات دائرية متحركة لعرض النسب المئوية.

```html
<div class="circular-progress" data-value="95">
    <!-- المحتوى -->
</div>
```

### 3. Training Charts
رسوم بيانية لتتبع Loss و Accuracy.

```javascript
createLossChart();          // رسم Loss
createTrainingMetricsChart(); // رسم Metrics متعددة
```

### 4. System Monitor
مراقبة النظام في الوقت الفعلي.

```javascript
const monitor = new SystemMonitor();
monitor.startMonitoring();
```

---

## 🎨 التصميم

### نظام الألوان
| اللون | Hex | الاستخدام |
|------|-----|----------|
| Primary Cyan | `#00d9ff` | العناصر الرئيسية |
| Primary Teal | `#00ffcc` | التمييزات |
| Accent Blue | `#0099ff` | الروابط والأزرار |
| Success Green | `#00ff88` | النجاح |
| Warning Yellow | `#ffd500` | التحذيرات |
| Error Red | `#ff3366` | الأخطاء |

### الخطوط
- **Display Font**: Orbitron (للعناوين)
- **Body Font**: Rajdhani (للنصوص)
- **Code Font**: Courier New (للأكواد)

---

## 🔧 الميزات التقنية

### Technologies Used
- **HTML5** - البنية الأساسية
- **CSS3** - التصميم والتأثيرات
  - CSS Grid & Flexbox
  - CSS Animations
  - CSS Variables
  - Backdrop Filter
- **JavaScript ES6+** - البرمجة
  - Canvas API
  - Chart.js v4
  - Classes & Modules
  - Async/Await
- **Chart.js** - الرسوم البيانية

### Browser Support
| Browser | Version |
|---------|---------|
| Chrome | 90+ |
| Firefox | 88+ |
| Safari | 14+ |
| Edge | 90+ |

---

## 🤝 المساهمة

نرحب بمساهماتكم! يمكنك:

1. Fork المشروع
2. إنشاء Branch جديد (`git checkout -b feature/AmazingFeature`)
3. Commit التغييرات (`git commit -m 'Add AmazingFeature'`)
4. Push إلى Branch (`git push origin feature/AmazingFeature`)
5. فتح Pull Request

---

## 📝 الترخيص

هذا المشروع مرخص تحت رخصة MIT - انظر ملف [LICENSE](LICENSE) للتفاصيل.

---

## 👨‍💻 المطور

**M.3R3**
- تصميم وتطوير واجهة AI Forge Studio
- خبرة في تطوير واجهات المستخدم المتقدمة
- متخصص في التصميمات Futuristic و Tech-themed

---

## 📞 التواصل

هل لديك سؤال أو اقتراح؟ تواصل معنا:

- 🌐 Website: [https://aiforgestudio.net](https://aiforgestudio.net)
- 💻 GitHub: [@QAZ83](https://github.com/QAZ83)
- 📚 Documentation: [CPP-Qt-Integration-Guide.md](CPP-Qt-Integration-Guide.md)

---

## 🙏 شكر خاص

- [Chart.js](https://www.chartjs.org/) - للرسوم البيانية الرائعة
- [Google Fonts](https://fonts.google.com/) - لخطوط Orbitron و Rajdhani
- المجتمع البرمجي العربي

---

<div align="center">

**صُنع بـ ❤️ في العالم العربي**

⭐ إذا أعجبك المشروع، لا تنسى إعطائه نجمة!

</div>