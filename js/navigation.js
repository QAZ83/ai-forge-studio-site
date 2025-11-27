/**
 * AI Forge Studio - Unified Navigation System
 * Handles page switching for both web and Electron modes
 * Author: M.3R3
 */

// ===================================
// Navigation Configuration
// ===================================
const NavigationConfig = {
    pages: {
        'index': { title: 'الرئيسية', icon: '🏠', file: 'index.html' },
        'dashboard': { title: 'لوحة التحكم', icon: '📊', file: 'dashboard.html' },
        'cuda': { title: 'CUDA', icon: '⚡', file: 'cuda-integration.html' },
        'tensorrt': { title: 'TensorRT', icon: '🧠', file: 'tensorrt.html' },
        'vulkan': { title: 'Vulkan', icon: '🎮', file: 'vulkan.html' },
        'inference': { title: 'Inference', icon: '🚀', file: 'inference.html' },
        'download': { title: 'تحميل', icon: '⬇️', file: 'download.html' },
        'multi-dashboard': { title: 'Multi Dashboard', icon: '📈', file: 'multi-dashboard.html' }
    },
    transitions: {
        duration: 300,
        easing: 'ease-out'
    }
};

// ===================================
// Navigation Class
// ===================================
class NavigationSystem {
    constructor() {
        this.isElectron = this.detectElectron();
        this.currentPage = this.getCurrentPage();
        this.transitionInProgress = false;
        
        this.init();
    }

    /**
     * Detect if running in Electron
     */
    detectElectron() {
        return typeof window !== 'undefined' && 
               typeof window.electronAPI !== 'undefined' && 
               window.electronAPI.isElectron === true;
    }

    /**
     * Get current page from URL
     */
    getCurrentPage() {
        const path = window.location.pathname;
        const filename = path.split('/').pop() || 'index.html';
        
        for (const [key, config] of Object.entries(NavigationConfig.pages)) {
            if (config.file === filename) {
                return key;
            }
        }
        return 'index';
    }

    /**
     * Initialize navigation system
     */
    init() {
        console.log(`🧭 Navigation System Initialized (${this.isElectron ? 'Electron' : 'Web'} mode)`);
        
        // Setup navigation buttons
        this.setupNavigationButtons();
        
        // Add page transition styles
        this.addTransitionStyles();
        
        // Handle back/forward browser navigation
        window.addEventListener('popstate', () => {
            this.currentPage = this.getCurrentPage();
            this.updateActiveState();
        });

        // Setup keyboard shortcuts for navigation
        this.setupKeyboardNavigation();
    }

    /**
     * Add CSS transition styles
     */
    addTransitionStyles() {
        if (document.getElementById('nav-transition-styles')) return;

        const style = document.createElement('style');
        style.id = 'nav-transition-styles';
        style.textContent = `
            .page-transition-out {
                animation: pageOut ${NavigationConfig.transitions.duration}ms ${NavigationConfig.transitions.easing};
            }
            
            .page-transition-in {
                animation: pageIn ${NavigationConfig.transitions.duration}ms ${NavigationConfig.transitions.easing};
            }
            
            @keyframes pageOut {
                from {
                    opacity: 1;
                    transform: translateY(0);
                }
                to {
                    opacity: 0;
                    transform: translateY(-20px);
                }
            }
            
            @keyframes pageIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .nav-btn {
                transition: all 0.3s ease;
            }
            
            .nav-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3);
            }
            
            .nav-btn.active {
                background: linear-gradient(135deg, rgba(0, 217, 255, 0.2), rgba(0, 153, 255, 0.1));
                border-color: #00d9ff;
            }
        `;
        document.head.appendChild(style);
    }

    /**
     * Setup all navigation buttons
     */
    setupNavigationButtons() {
        // Select all navigation buttons (including .nav-icon-btn for multi-dashboard)
        const navButtons = document.querySelectorAll('.nav-btn[data-page], .nav-btn[onclick*="location.href"], .nav-icon-btn[data-page], .nav-icon-btn[onclick*="location.href"]');
        
        navButtons.forEach(button => {
            // Extract page from onclick if present
            const onclick = button.getAttribute('onclick');
            if (onclick && onclick.includes('location.href')) {
                const match = onclick.match(/['"]([^'"]+\.html)['"]/);
                if (match) {
                    const page = this.getPageKeyFromFile(match[1]);
                    if (page) {
                        button.setAttribute('data-page', page);
                        button.removeAttribute('onclick');
                    }
                }
            }
            
            // Add click handler
            const page = button.getAttribute('data-page');
            if (page) {
                button.addEventListener('click', (e) => {
                    e.preventDefault();
                    this.navigateTo(page);
                });
            }
        });

        // Update active state
        this.updateActiveState();
    }

    /**
     * Get page key from filename
     */
    getPageKeyFromFile(filename) {
        for (const [key, config] of Object.entries(NavigationConfig.pages)) {
            if (config.file === filename) {
                return key;
            }
        }
        return null;
    }

    /**
     * Navigate to a page
     */
    navigateTo(pageKey) {
        if (this.transitionInProgress) return;
        if (pageKey === this.currentPage) return;

        const pageConfig = NavigationConfig.pages[pageKey];
        if (!pageConfig) {
            console.error(`Unknown page: ${pageKey}`);
            return;
        }

        this.transitionInProgress = true;
        
        // Apply exit transition
        const mainContainer = document.querySelector('.main-container');
        if (mainContainer) {
            mainContainer.classList.add('page-transition-out');
        }

        // Navigate after transition
        setTimeout(() => {
            if (this.isElectron) {
                // In Electron, use file protocol
                window.location.href = pageConfig.file;
            } else {
                // In web, use relative URL
                window.location.href = pageConfig.file;
            }
        }, NavigationConfig.transitions.duration / 2);
    }

    /**
     * Update active navigation state
     */
    updateActiveState() {
        const navButtons = document.querySelectorAll('.nav-btn, .nav-icon-btn');
        
        navButtons.forEach(button => {
            button.classList.remove('active');
            
            const page = button.getAttribute('data-page');
            if (page === this.currentPage) {
                button.classList.add('active');
            }
            
            // Also check onclick attribute for legacy support
            const onclick = button.getAttribute('onclick');
            if (onclick) {
                const currentFile = NavigationConfig.pages[this.currentPage]?.file;
                if (currentFile && onclick.includes(currentFile)) {
                    button.classList.add('active');
                }
            }
        });
    }

    /**
     * Setup keyboard navigation shortcuts
     */
    setupKeyboardNavigation() {
        document.addEventListener('keydown', (e) => {
            // Alt + number for quick navigation
            if (e.altKey && !e.ctrlKey && !e.shiftKey) {
                const pageMap = {
                    '1': 'index',
                    '2': 'dashboard',
                    '3': 'cuda',
                    '4': 'tensorrt',
                    '5': 'vulkan',
                    '6': 'inference'
                };
                
                const page = pageMap[e.key];
                if (page) {
                    e.preventDefault();
                    this.navigateTo(page);
                }
            }
        });
    }

    /**
     * Get navigation HTML for injecting into pages
     */
    static getNavigationHTML(activePage = 'index') {
        let html = '';
        const mainPages = ['index', 'dashboard', 'cuda', 'tensorrt', 'vulkan', 'inference'];
        
        mainPages.forEach(pageKey => {
            const config = NavigationConfig.pages[pageKey];
            const isActive = pageKey === activePage ? 'active' : '';
            html += `
                <button class="nav-btn ${isActive}" data-page="${pageKey}">
                    <span class="icon">${config.icon}</span>
                    <span>${config.title}</span>
                </button>
            `;
        });
        
        return html;
    }
}

// ===================================
// Page Load Handler
// ===================================
class PageLoadHandler {
    constructor() {
        this.init();
    }

    init() {
        // Add entry animation
        document.addEventListener('DOMContentLoaded', () => {
            const mainContainer = document.querySelector('.main-container');
            if (mainContainer) {
                mainContainer.classList.add('page-transition-in');
                
                // Remove class after animation
                setTimeout(() => {
                    mainContainer.classList.remove('page-transition-in');
                }, NavigationConfig.transitions.duration);
            }
        });

        // Handle visibility change (tab switching)
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible') {
                this.onPageVisible();
            }
        });
    }

    onPageVisible() {
        // Resume animations if paused
        console.log('📄 Page is now visible');
    }
}

// ===================================
// Error Handler for Navigation
// ===================================
class NavigationErrorHandler {
    constructor() {
        this.init();
    }

    init() {
        window.addEventListener('error', (e) => {
            if (e.message.includes('navigation') || e.message.includes('location')) {
                console.error('Navigation error:', e.message);
                this.handleNavigationError(e);
            }
        });
    }

    handleNavigationError(error) {
        // Show error notification if notify is available
        if (typeof notify !== 'undefined') {
            notify.show('Navigation error occurred. Please try again.', 'error');
        }
    }
}

// ===================================
// Initialize on Load
// ===================================
let navigation;
let pageLoadHandler;
let navErrorHandler;

document.addEventListener('DOMContentLoaded', function() {
    navigation = new NavigationSystem();
    pageLoadHandler = new PageLoadHandler();
    navErrorHandler = new NavigationErrorHandler();
});

// Export for use in other scripts
window.NavigationSystem = NavigationSystem;
window.NavigationConfig = NavigationConfig;
