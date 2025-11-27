/**
 * AI Forge Studio - Internationalization (i18n)
 * Multi-language Support System
 * Designed by: M.3R3
 */

const translations = {
    ar: {
        // Navigation
        'home': 'الرئيسية',
        'dashboard': 'لوحة التحكم',
        'load': 'تحميل',
        'terminal': 'الطرفية',
        'search': 'بحث',

        // Main Sections
        'gpu_output': 'مخرجات تسريع GPU',
        'system_status': 'حالة النظام',
        'training_experimentation': 'التدريب والتجريب',
        'development_environment': 'بيئة التطوير',
        'code_editor': 'محرر الأكواد',
        'performance_dashboard': 'لوحة الأداء',

        // GPU Status
        'current_gpu_usage': 'استخدام GPU الحالي',
        'cpu_usage': 'استخدام المعالج',
        'memory': 'الذاكرة',
        'temperature': 'درجة الحرارة',
        'inference_speed': 'سرعة الاستدلال',

        // Training
        'loss_reductions': 'تقليل الخسائر',
        'visualize_web_output': 'عرض المخرجات الويب',
        'local': 'محلي',
        'generate_to_cloudflare': 'إنشاء على Cloudflare.net',

        // Model Manager
        'model_manager': 'مدير النماذج',
        'loaded_models': 'النماذج المحملة',
        'upload': 'رفع',
        'run': 'تشغيل',
        'loading': 'جار التحميل',
        'model_conversion': 'تحويل النموذج',
        'convert_to_tensorrt': 'تحويل إلى TensorRT',

        // Status
        'online': 'متصل',
        'offline': 'غير متصل',
        'running': 'قيد التشغيل',
        'pending': 'قيد الانتظار',
        'active': 'نشط',

        // API & Cloud
        'cloud_api_interface': 'واجهة السحابة / API',
        'cloudflare_workers_status': 'حالة Cloudflare Workers',
        'api_endpoint_health': 'صحة نقاط API',
        'deploying_to_cloudflare': 'النشر على Cloudflare',

        // Download Page
        'download_app': 'تحميل التطبيق',
        'download_for_windows': 'تحميل لنظام Windows',
        'download_for_mac': 'تحميل لنظام macOS',
        'download_for_linux': 'تحميل لنظام Linux',
        'web_version': 'النسخة الإلكترونية',
        'system_requirements': 'متطلبات النظام',

        // Settings
        'language': 'اللغة',
        'arabic': 'العربية',
        'english': 'English',
        'theme': 'المظهر',
        'dark_mode': 'الوضع الداكن',
        'light_mode': 'الوضع الفاتح',

        // Tooltips & Messages
        'switch_language': 'تبديل اللغة',
        'building_project': 'جاري بناء المشروع...',
        'build_completed': 'اكتمل البناء بنجاح',
        'ready_for_inference': 'جاهز للاستدلال',

        // TensorRT Specific
        'tensorrt_integration': 'تكامل TensorRT',
        'tensorrt_engine_status': 'حالة محرك TensorRT',
        'precision_mode': 'وضع الدقة',
        'tensor_cores': 'أنوية Tensor',
        'optimization_tips': 'نصائح التحسين',

        // Training Config
        'training_configs': 'إعدادات التدريب',
        'experimentation': 'التجريب',
        'hypertuning': 'ضبط المعاملات الفائقة',
        'training_mode': 'وضع التدريب',
        'enhanced_mode': 'الوضع المحسّن',
        'tracking_tester': 'مُختبر التتبع',
        'override': 'تجاوز',
        'batch_size': 'حجم الدفعة',

        // Performance
        'gpu_tracking_constantly': 'تتبع GPU مستمر',
        'status_tracking_constantly': 'تتبع الحالة مستمر',
        'training_loss_metrics': 'مقاييس خسائر التدريب',
        'ai_training_studio': 'استوديو تدريب الذكاء الاصطناعي',

        // Developer
        'built_by': 'من بناء',
        'version': 'الإصدار',
        'documentation': 'التوثيق',
        'support': 'الدعم',
        'about': 'حول',
    },

    en: {
        // Navigation
        'home': 'Home',
        'dashboard': 'Dashboard',
        'load': 'Load',
        'terminal': 'Terminal',
        'search': 'Search',

        // Main Sections
        'gpu_output': 'GPU Accelerated Output',
        'system_status': 'System Status',
        'training_experimentation': 'Training & Experimentation',
        'development_environment': 'Development Environment',
        'code_editor': 'Code Editor',
        'performance_dashboard': 'Performance Dashboard',

        // GPU Status
        'current_gpu_usage': 'Current GPU Usage',
        'cpu_usage': 'CPU Usage',
        'memory': 'Memory',
        'temperature': 'Temperature',
        'inference_speed': 'Inference Speed',

        // Training
        'loss_reductions': 'Loss & Reductions',
        'visualize_web_output': 'Visualize Web Output',
        'local': 'Local',
        'generate_to_cloudflare': 'Generate to Cloudflare.net',

        // Model Manager
        'model_manager': 'Model Manager',
        'loaded_models': 'Loaded Models',
        'upload': 'Upload',
        'run': 'Run',
        'loading': 'Loading',
        'model_conversion': 'Model Conversion',
        'convert_to_tensorrt': 'Convert to TensorRT',

        // Status
        'online': 'Online',
        'offline': 'Offline',
        'running': 'Running',
        'pending': 'Pending',
        'active': 'Active',

        // API & Cloud
        'cloud_api_interface': 'Cloud / API Interface',
        'cloudflare_workers_status': 'Cloudflare Workers Status',
        'api_endpoint_health': 'API Endpoint Health',
        'deploying_to_cloudflare': 'Deploying to Cloudflare',

        // Download Page
        'download_app': 'Download Application',
        'download_for_windows': 'Download for Windows',
        'download_for_mac': 'Download for macOS',
        'download_for_linux': 'Download for Linux',
        'web_version': 'Web Version',
        'system_requirements': 'System Requirements',

        // Settings
        'language': 'Language',
        'arabic': 'العربية',
        'english': 'English',
        'theme': 'Theme',
        'dark_mode': 'Dark Mode',
        'light_mode': 'Light Mode',

        // Tooltips & Messages
        'switch_language': 'Switch Language',
        'building_project': 'Building project...',
        'build_completed': 'Build completed successfully',
        'ready_for_inference': 'Ready for AI Inference',

        // TensorRT Specific
        'tensorrt_integration': 'TensorRT Integration',
        'tensorrt_engine_status': 'TensorRT Engine Status',
        'precision_mode': 'Precision Mode',
        'tensor_cores': 'Tensor Cores',
        'optimization_tips': 'Optimization Tips',

        // Training Config
        'training_configs': 'Training Configs',
        'experimentation': 'Experimentation',
        'hypertuning': 'Hypertuning',
        'training_mode': 'Training Mode',
        'enhanced_mode': 'Enhanced Mode',
        'tracking_tester': 'Tracking Tester',
        'override': 'Override',
        'batch_size': 'Batch Size',

        // Performance
        'gpu_tracking_constantly': 'GPU Tracking Constantly',
        'status_tracking_constantly': 'Status Tracking Constantly',
        'training_loss_metrics': 'Training Loss & Metrics',
        'ai_training_studio': 'AI Training Studio',

        // Developer
        'built_by': 'Built by',
        'version': 'Version',
        'documentation': 'Documentation',
        'support': 'Support',
        'about': 'About',
    }
};

// Current language state
let currentLanguage = localStorage.getItem('aiforge_language') || 'ar';

// Translation function
function t(key) {
    return translations[currentLanguage][key] || key;
}

// Change language
function changeLanguage(lang) {
    currentLanguage = lang;
    localStorage.setItem('aiforge_language', lang);

    // Update HTML attributes
    document.documentElement.lang = lang;
    document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr';

    // Update all translatable elements
    updateTranslations();
}

// Update all elements with data-i18n attribute
function updateTranslations() {
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        element.textContent = t(key);
    });

    // Update placeholders
    document.querySelectorAll('[data-i18n-placeholder]').forEach(element => {
        const key = element.getAttribute('data-i18n-placeholder');
        element.placeholder = t(key);
    });

    // Update titles
    document.querySelectorAll('[data-i18n-title]').forEach(element => {
        const key = element.getAttribute('data-i18n-title');
        element.title = t(key);
    });
}

// Create language switcher UI
function createLanguageSwitcher() {
    const switcher = document.createElement('button');
    switcher.className = 'language-switcher';
    switcher.innerHTML = `
        <span class="icon">🌐</span>
        <span data-i18n="language">${t('language')}</span>
    `;
    switcher.onclick = toggleLanguage;

    return switcher;
}

// Toggle between languages
function toggleLanguage() {
    const newLang = currentLanguage === 'ar' ? 'en' : 'ar';
    changeLanguage(newLang);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Set initial language
    changeLanguage(currentLanguage);

    // Add language switcher to nav
    const nav = document.querySelector('.main-nav');
    if (nav) {
        const switcher = createLanguageSwitcher();
        nav.insertBefore(switcher, nav.firstChild);
    }
});

// Export functions for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { t, changeLanguage, updateTranslations };
}
