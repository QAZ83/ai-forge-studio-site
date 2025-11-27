/**
 * AI Forge Studio - Qt Main Application
 * Author: M.3R3
 * 
 * Qt6 entry point with GPU Performance Panel integration.
 */

#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QProgressBar>
#include <QFrame>
#include <QTimer>
#include <QPushButton>
#include <QStackedWidget>
#include <QListWidget>
#include <QTableWidget>
#include <QHeaderView>
#include <QGroupBox>
#include <QSplitter>
#include <QStatusBar>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QMessageBox>
#include <QFont>
#include <QPalette>
#include <QStyleFactory>
#include <QScreen>
#include <QChartView>
#include <QLineSeries>
#include <QChart>
#include <QValueAxis>
#include <QSplineSeries>
#include <QAreaSeries>

#include "src/gpu/cuda_inference.h"
#include "src/gpu/tensorrt_engine.h"
#include "src/gpu/vulkan_renderer.h"
#include "src/monitor/gpu_monitor.h"

using namespace aiforge;
using namespace QtCharts;

//=============================================================================
// GPU Performance Panel Widget
//=============================================================================

class GPUPerformancePanel : public QWidget {
    Q_OBJECT

public:
    explicit GPUPerformancePanel(QWidget* parent = nullptr)
        : QWidget(parent)
        , m_dataPoints(0)
    {
        setupUI();
        setupCharts();
        
        // Initialize monitor
        m_monitor = std::make_unique<monitor::GPUMonitor>();
        if (m_monitor->initialize()) {
            // Start 100ms update timer
            m_updateTimer = new QTimer(this);
            connect(m_updateTimer, &QTimer::timeout, this, &GPUPerformancePanel::updateMetrics);
            m_updateTimer->start(100);
        }
    }

    ~GPUPerformancePanel() override {
        if (m_updateTimer) {
            m_updateTimer->stop();
        }
    }

private slots:
    void updateMetrics() {
        if (!m_monitor || !m_monitor->isReady()) {
            return;
        }

        auto metrics = m_monitor->getMetrics(0);
        
        // Update labels
        m_gpuNameLabel->setText(QString::fromStdString(metrics.name));
        m_gpuUsageLabel->setText(QString("%1%").arg(metrics.gpuUsagePercent, 0, 'f', 1));
        m_memoryUsageLabel->setText(QString("%1 / %2 MB")
            .arg(metrics.memoryUsedMB)
            .arg(metrics.memoryTotalMB));
        m_temperatureLabel->setText(QString("%1°C").arg(metrics.temperatureCelsius, 0, 'f', 0));
        m_powerLabel->setText(QString("%1W / %2W")
            .arg(metrics.powerDrawWatts, 0, 'f', 0)
            .arg(metrics.powerLimitWatts, 0, 'f', 0));
        m_clockLabel->setText(QString("%1 MHz").arg(metrics.clockSpeedMHz));
        m_memClockLabel->setText(QString("%1 MHz").arg(metrics.memoryClockMHz));
        m_fanLabel->setText(QString("%1%").arg(metrics.fanSpeedPercent));
        m_pcieLabel->setText(QString("PCIe %1 x%2 | TX: %3 MB/s | RX: %4 MB/s")
            .arg(metrics.pcieGen)
            .arg(metrics.pcieLanes)
            .arg(metrics.pcieTxMBps)
            .arg(metrics.pcieRxMBps));
        
        // Update progress bars
        m_gpuUsageBar->setValue(static_cast<int>(metrics.gpuUsagePercent));
        m_memoryUsageBar->setValue(static_cast<int>(metrics.memoryUsagePercent));
        m_temperatureBar->setValue(static_cast<int>(metrics.temperatureCelsius));
        m_powerBar->setValue(static_cast<int>(
            (metrics.powerDrawWatts / metrics.powerLimitWatts) * 100));

        // Update charts
        updateCharts(metrics);
        
        // Status color
        if (metrics.temperatureCelsius > 80) {
            m_temperatureLabel->setStyleSheet("color: #ef4444;");
        } else if (metrics.temperatureCelsius > 70) {
            m_temperatureLabel->setStyleSheet("color: #f59e0b;");
        } else {
            m_temperatureLabel->setStyleSheet("color: #10b981;");
        }
    }

private:
    void setupUI() {
        auto* mainLayout = new QVBoxLayout(this);
        mainLayout->setContentsMargins(16, 16, 16, 16);
        mainLayout->setSpacing(16);

        // Header
        auto* headerLayout = new QHBoxLayout();
        m_gpuNameLabel = new QLabel("Detecting GPU...");
        m_gpuNameLabel->setStyleSheet("font-size: 18px; font-weight: bold; color: #e2e8f0;");
        headerLayout->addWidget(m_gpuNameLabel);
        headerLayout->addStretch();
        mainLayout->addLayout(headerLayout);

        // Main content area with splitter
        auto* splitter = new QSplitter(Qt::Horizontal);
        splitter->setChildrenCollapsible(false);

        // Left panel - Metrics
        auto* metricsWidget = new QWidget();
        auto* metricsLayout = new QVBoxLayout(metricsWidget);
        metricsLayout->setContentsMargins(0, 0, 0, 0);

        // GPU Usage
        metricsLayout->addWidget(createMetricWidget("GPU Usage", 
            m_gpuUsageLabel = new QLabel("0%"),
            m_gpuUsageBar = new QProgressBar()));
        
        // Memory Usage
        metricsLayout->addWidget(createMetricWidget("VRAM", 
            m_memoryUsageLabel = new QLabel("0 / 0 MB"),
            m_memoryUsageBar = new QProgressBar()));

        // Temperature
        m_temperatureBar = new QProgressBar();
        m_temperatureBar->setRange(0, 100);
        metricsLayout->addWidget(createMetricWidget("Temperature", 
            m_temperatureLabel = new QLabel("0°C"),
            m_temperatureBar));

        // Power
        m_powerBar = new QProgressBar();
        m_powerBar->setRange(0, 100);
        metricsLayout->addWidget(createMetricWidget("Power Draw", 
            m_powerLabel = new QLabel("0W / 0W"),
            m_powerBar));

        // Clocks
        auto* clocksGroup = new QGroupBox("Clock Speeds");
        clocksGroup->setStyleSheet(groupBoxStyle());
        auto* clocksLayout = new QGridLayout(clocksGroup);
        clocksLayout->addWidget(new QLabel("Core:"), 0, 0);
        m_clockLabel = new QLabel("0 MHz");
        clocksLayout->addWidget(m_clockLabel, 0, 1);
        clocksLayout->addWidget(new QLabel("Memory:"), 1, 0);
        m_memClockLabel = new QLabel("0 MHz");
        clocksLayout->addWidget(m_memClockLabel, 1, 1);
        clocksLayout->addWidget(new QLabel("Fan:"), 2, 0);
        m_fanLabel = new QLabel("0%");
        clocksLayout->addWidget(m_fanLabel, 2, 1);
        metricsLayout->addWidget(clocksGroup);

        // PCIe Info
        m_pcieLabel = new QLabel("PCIe - x0");
        m_pcieLabel->setStyleSheet("color: #94a3b8; font-size: 12px;");
        metricsLayout->addWidget(m_pcieLabel);

        metricsLayout->addStretch();
        splitter->addWidget(metricsWidget);

        // Right panel - Charts
        auto* chartsWidget = new QWidget();
        auto* chartsLayout = new QVBoxLayout(chartsWidget);
        chartsLayout->setContentsMargins(0, 0, 0, 0);

        m_gpuChartView = new QChartView();
        m_gpuChartView->setRenderHint(QPainter::Antialiasing);
        m_gpuChartView->setMinimumHeight(150);
        chartsLayout->addWidget(new QLabel("GPU Utilization"));
        chartsLayout->addWidget(m_gpuChartView);

        m_memChartView = new QChartView();
        m_memChartView->setRenderHint(QPainter::Antialiasing);
        m_memChartView->setMinimumHeight(150);
        chartsLayout->addWidget(new QLabel("Memory Usage"));
        chartsLayout->addWidget(m_memChartView);

        m_tempChartView = new QChartView();
        m_tempChartView->setRenderHint(QPainter::Antialiasing);
        m_tempChartView->setMinimumHeight(150);
        chartsLayout->addWidget(new QLabel("Temperature"));
        chartsLayout->addWidget(m_tempChartView);

        splitter->addWidget(chartsWidget);
        splitter->setSizes({350, 500});

        mainLayout->addWidget(splitter);
    }

    void setupCharts() {
        // GPU Usage Chart
        m_gpuSeries = new QSplineSeries();
        m_gpuSeries->setColor(QColor("#3b82f6"));
        
        auto* gpuChart = new QChart();
        gpuChart->addSeries(m_gpuSeries);
        gpuChart->setBackgroundBrush(QBrush(QColor("#1e293b")));
        gpuChart->legend()->hide();
        gpuChart->setMargins(QMargins(0, 0, 0, 0));
        
        auto* gpuAxisX = new QValueAxis();
        gpuAxisX->setRange(0, 60);
        gpuAxisX->setLabelsVisible(false);
        gpuAxisX->setGridLineColor(QColor("#334155"));
        
        auto* gpuAxisY = new QValueAxis();
        gpuAxisY->setRange(0, 100);
        gpuAxisY->setLabelFormat("%d%%");
        gpuAxisY->setLabelsColor(QColor("#94a3b8"));
        gpuAxisY->setGridLineColor(QColor("#334155"));
        
        gpuChart->addAxis(gpuAxisX, Qt::AlignBottom);
        gpuChart->addAxis(gpuAxisY, Qt::AlignLeft);
        m_gpuSeries->attachAxis(gpuAxisX);
        m_gpuSeries->attachAxis(gpuAxisY);
        
        m_gpuChartView->setChart(gpuChart);

        // Memory Usage Chart
        m_memSeries = new QSplineSeries();
        m_memSeries->setColor(QColor("#8b5cf6"));
        
        auto* memChart = new QChart();
        memChart->addSeries(m_memSeries);
        memChart->setBackgroundBrush(QBrush(QColor("#1e293b")));
        memChart->legend()->hide();
        memChart->setMargins(QMargins(0, 0, 0, 0));
        
        auto* memAxisX = new QValueAxis();
        memAxisX->setRange(0, 60);
        memAxisX->setLabelsVisible(false);
        memAxisX->setGridLineColor(QColor("#334155"));
        
        auto* memAxisY = new QValueAxis();
        memAxisY->setRange(0, 100);
        memAxisY->setLabelFormat("%d%%");
        memAxisY->setLabelsColor(QColor("#94a3b8"));
        memAxisY->setGridLineColor(QColor("#334155"));
        
        memChart->addAxis(memAxisX, Qt::AlignBottom);
        memChart->addAxis(memAxisY, Qt::AlignLeft);
        m_memSeries->attachAxis(memAxisX);
        m_memSeries->attachAxis(memAxisY);
        
        m_memChartView->setChart(memChart);

        // Temperature Chart
        m_tempSeries = new QSplineSeries();
        m_tempSeries->setColor(QColor("#10b981"));
        
        auto* tempChart = new QChart();
        tempChart->addSeries(m_tempSeries);
        tempChart->setBackgroundBrush(QBrush(QColor("#1e293b")));
        tempChart->legend()->hide();
        tempChart->setMargins(QMargins(0, 0, 0, 0));
        
        auto* tempAxisX = new QValueAxis();
        tempAxisX->setRange(0, 60);
        tempAxisX->setLabelsVisible(false);
        tempAxisX->setGridLineColor(QColor("#334155"));
        
        auto* tempAxisY = new QValueAxis();
        tempAxisY->setRange(0, 100);
        tempAxisY->setLabelFormat("%d°C");
        tempAxisY->setLabelsColor(QColor("#94a3b8"));
        tempAxisY->setGridLineColor(QColor("#334155"));
        
        tempChart->addAxis(tempAxisX, Qt::AlignBottom);
        tempChart->addAxis(tempAxisY, Qt::AlignLeft);
        m_tempSeries->attachAxis(tempAxisX);
        m_tempSeries->attachAxis(tempAxisY);
        
        m_tempChartView->setChart(tempChart);
    }

    void updateCharts(const monitor::GPUMetrics& metrics) {
        const int maxPoints = 60;
        
        // Add new data points
        m_gpuSeries->append(m_dataPoints, metrics.gpuUsagePercent);
        m_memSeries->append(m_dataPoints, metrics.memoryUsagePercent);
        m_tempSeries->append(m_dataPoints, metrics.temperatureCelsius);
        
        m_dataPoints++;
        
        // Remove old points and adjust axis
        if (m_gpuSeries->count() > maxPoints) {
            m_gpuSeries->remove(0);
            m_memSeries->remove(0);
            m_tempSeries->remove(0);
            
            // Adjust X axis
            auto updateAxis = [this, maxPoints](QChartView* view) {
                auto axes = view->chart()->axes(Qt::Horizontal);
                if (!axes.isEmpty()) {
                    static_cast<QValueAxis*>(axes.first())->setRange(
                        m_dataPoints - maxPoints, m_dataPoints);
                }
            };
            updateAxis(m_gpuChartView);
            updateAxis(m_memChartView);
            updateAxis(m_tempChartView);
        }
    }

    QWidget* createMetricWidget(const QString& title, QLabel* valueLabel, 
                                 QProgressBar* progressBar) {
        auto* widget = new QFrame();
        widget->setStyleSheet(
            "QFrame { background: #1e293b; border-radius: 8px; padding: 12px; }");
        
        auto* layout = new QVBoxLayout(widget);
        layout->setContentsMargins(12, 8, 12, 8);
        layout->setSpacing(4);
        
        auto* headerLayout = new QHBoxLayout();
        auto* titleLabel = new QLabel(title);
        titleLabel->setStyleSheet("color: #94a3b8; font-size: 12px;");
        headerLayout->addWidget(titleLabel);
        headerLayout->addStretch();
        valueLabel->setStyleSheet("color: #e2e8f0; font-size: 14px; font-weight: bold;");
        headerLayout->addWidget(valueLabel);
        layout->addLayout(headerLayout);
        
        progressBar->setRange(0, 100);
        progressBar->setTextVisible(false);
        progressBar->setFixedHeight(6);
        progressBar->setStyleSheet(
            "QProgressBar { background: #334155; border-radius: 3px; }"
            "QProgressBar::chunk { background: #3b82f6; border-radius: 3px; }");
        layout->addWidget(progressBar);
        
        return widget;
    }

    QString groupBoxStyle() const {
        return "QGroupBox { "
               "  color: #e2e8f0; "
               "  font-weight: bold; "
               "  border: 1px solid #334155; "
               "  border-radius: 8px; "
               "  margin-top: 8px; "
               "  padding-top: 8px; "
               "} "
               "QGroupBox::title { "
               "  subcontrol-origin: margin; "
               "  left: 10px; "
               "}";
    }

private:
    std::unique_ptr<monitor::GPUMonitor> m_monitor;
    QTimer* m_updateTimer = nullptr;
    int m_dataPoints;

    // Labels
    QLabel* m_gpuNameLabel;
    QLabel* m_gpuUsageLabel;
    QLabel* m_memoryUsageLabel;
    QLabel* m_temperatureLabel;
    QLabel* m_powerLabel;
    QLabel* m_clockLabel;
    QLabel* m_memClockLabel;
    QLabel* m_fanLabel;
    QLabel* m_pcieLabel;

    // Progress bars
    QProgressBar* m_gpuUsageBar;
    QProgressBar* m_memoryUsageBar;
    QProgressBar* m_temperatureBar;
    QProgressBar* m_powerBar;

    // Charts
    QChartView* m_gpuChartView;
    QChartView* m_memChartView;
    QChartView* m_tempChartView;
    QSplineSeries* m_gpuSeries;
    QSplineSeries* m_memSeries;
    QSplineSeries* m_tempSeries;
};

//=============================================================================
// Inference Control Panel
//=============================================================================

class InferencePanel : public QWidget {
    Q_OBJECT

public:
    explicit InferencePanel(QWidget* parent = nullptr) : QWidget(parent) {
        setupUI();
        m_tensorrtEngine = std::make_unique<gpu::TensorRTEngine>();
    }

private slots:
    void runCudaBenchmark() {
        m_statusLabel->setText("Running CUDA benchmark...");
        
        gpu::CudaInference cuda;
        if (!cuda.initialize()) {
            m_statusLabel->setText("CUDA initialization failed!");
            return;
        }
        
        float ms = cuda.runBenchmark();
        m_statusLabel->setText(QString("CUDA Benchmark: %1 ms").arg(ms, 0, 'f', 2));
    }

    void loadTensorRTModel() {
        m_statusLabel->setText("Loading TensorRT engine...");
        
        // Example: Build from ONNX or load from serialized engine
        gpu::TensorRTConfig config;
        config.modelPath = "model.onnx";  // Update with actual model path
        config.engineName = "demo_engine";
        config.fp16Mode = true;
        config.enableCudaGraphs = true;
        config.enablePreAllocatedOutputs = true;
        
        // Try to load pre-built engine first, then fall back to ONNX
        if (m_tensorrtEngine->loadEngine("model.engine")) {
            m_statusLabel->setText("TensorRT engine loaded from cache!");
            updateEngineInfo();
        } else if (m_tensorrtEngine->buildFromONNX(config)) {
            m_statusLabel->setText("TensorRT engine built from ONNX!");
            m_tensorrtEngine->saveEngine("model.engine");
            updateEngineInfo();
        } else {
            m_statusLabel->setText(QString("TensorRT error: %1")
                .arg(QString::fromStdString(m_tensorrtEngine->getLastError())));
        }
    }

    void runTensorRTBenchmark() {
        if (!m_tensorrtEngine->isReady()) {
            m_statusLabel->setText("Load a TensorRT engine first!");
            return;
        }
        
        m_statusLabel->setText("Running TensorRT benchmark...");
        float avgMs = m_tensorrtEngine->benchmark(100, 10);
        m_statusLabel->setText(QString("TensorRT Benchmark: %1 ms avg (%2 inferences/sec)")
            .arg(avgMs, 0, 'f', 3)
            .arg(1000.0f / avgMs, 0, 'f', 1));
    }

    void toggleCudaGraphs() {
        if (!m_tensorrtEngine->isReady()) return;
        
        bool enabled = !m_tensorrtEngine->getCudaGraphsEnabled();
        m_tensorrtEngine->setCudaGraphsEnabled(enabled);
        m_cudaGraphBtn->setText(QString("CUDA Graphs: %1").arg(enabled ? "ON" : "OFF"));
        m_statusLabel->setText(QString("CUDA Graphs %1").arg(enabled ? "enabled" : "disabled"));
    }

    void initVulkan() {
        m_statusLabel->setText("Initializing Vulkan...");
        
        gpu::VulkanRenderer renderer;
        if (!renderer.initialize(nullptr)) {  // Headless mode
            m_statusLabel->setText("Vulkan initialization failed!");
            return;
        }
        
        m_statusLabel->setText("Vulkan initialized successfully!");
    }

private:
    void setupUI() {
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(16, 16, 16, 16);
        layout->setSpacing(12);

        auto* titleLabel = new QLabel("Inference Controls");
        titleLabel->setStyleSheet("font-size: 18px; font-weight: bold; color: #e2e8f0;");
        layout->addWidget(titleLabel);

        // TensorRT Section
        auto* trtGroup = new QGroupBox("TensorRT Engine");
        trtGroup->setStyleSheet(groupBoxStyle());
        auto* trtLayout = new QVBoxLayout(trtGroup);
        
        auto* loadBtn = new QPushButton("Load TensorRT Engine");
        loadBtn->setStyleSheet(buttonStyle("#8b5cf6"));
        connect(loadBtn, &QPushButton::clicked, this, &InferencePanel::loadTensorRTModel);
        trtLayout->addWidget(loadBtn);
        
        auto* benchBtn = new QPushButton("Run TensorRT Benchmark");
        benchBtn->setStyleSheet(buttonStyle("#10b981"));
        connect(benchBtn, &QPushButton::clicked, this, &InferencePanel::runTensorRTBenchmark);
        trtLayout->addWidget(benchBtn);
        
        m_cudaGraphBtn = new QPushButton("CUDA Graphs: OFF");
        m_cudaGraphBtn->setStyleSheet(buttonStyle("#f59e0b"));
        connect(m_cudaGraphBtn, &QPushButton::clicked, this, &InferencePanel::toggleCudaGraphs);
        trtLayout->addWidget(m_cudaGraphBtn);
        
        m_engineInfoLabel = new QLabel("No engine loaded");
        m_engineInfoLabel->setStyleSheet("color: #94a3b8; font-family: monospace; font-size: 11px;");
        m_engineInfoLabel->setWordWrap(true);
        trtLayout->addWidget(m_engineInfoLabel);
        
        layout->addWidget(trtGroup);

        // CUDA Section
        auto* cudaBtn = new QPushButton("Run CUDA Benchmark");
        cudaBtn->setStyleSheet(buttonStyle("#3b82f6"));
        connect(cudaBtn, &QPushButton::clicked, this, &InferencePanel::runCudaBenchmark);
        layout->addWidget(cudaBtn);

        // Vulkan Section
        auto* vkBtn = new QPushButton("Initialize Vulkan");
        vkBtn->setStyleSheet(buttonStyle("#ec4899"));
        connect(vkBtn, &QPushButton::clicked, this, &InferencePanel::initVulkan);
        layout->addWidget(vkBtn);

        m_statusLabel = new QLabel("Ready");
        m_statusLabel->setStyleSheet("color: #94a3b8; padding: 8px; background: #1e293b; border-radius: 4px;");
        layout->addWidget(m_statusLabel);

        layout->addStretch();
    }
    
    void updateEngineInfo() {
        if (m_tensorrtEngine && m_tensorrtEngine->isReady()) {
            m_engineInfoLabel->setText(QString::fromStdString(m_tensorrtEngine->getEngineInfo()));
            m_cudaGraphBtn->setText(QString("CUDA Graphs: %1")
                .arg(m_tensorrtEngine->getCudaGraphsEnabled() ? "ON" : "OFF"));
        }
    }

    QString buttonStyle(const QString& color) const {
        return QString(
            "QPushButton {"
            "  background: %1; color: white; border: none; border-radius: 6px;"
            "  padding: 12px 24px; font-size: 14px; font-weight: bold;"
            "}"
            "QPushButton:hover { background: %2; }"
            "QPushButton:pressed { background: %3; }"
        ).arg(color)
         .arg(QColor(color).lighter(110).name())
         .arg(QColor(color).darker(110).name());
    }
    
    QString groupBoxStyle() const {
        return "QGroupBox { "
               "  color: #e2e8f0; "
               "  font-weight: bold; "
               "  border: 1px solid #334155; "
               "  border-radius: 8px; "
               "  margin-top: 8px; "
               "  padding: 12px; "
               "  padding-top: 24px; "
               "} "
               "QGroupBox::title { "
               "  subcontrol-origin: margin; "
               "  left: 10px; "
               "}";
    }

    QLabel* m_statusLabel;
    QLabel* m_engineInfoLabel;
    QPushButton* m_cudaGraphBtn;
    std::unique_ptr<gpu::TensorRTEngine> m_tensorrtEngine;
};

//=============================================================================
// Main Window
//=============================================================================

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr) : QMainWindow(parent) {
        setWindowTitle("AI Forge Studio - GPU Accelerated");
        setMinimumSize(1200, 800);
        
        setupDarkTheme();
        setupUI();
        setupMenus();
        
        statusBar()->showMessage("Ready");
    }

private:
    void setupDarkTheme() {
        QPalette palette;
        palette.setColor(QPalette::Window, QColor("#0f172a"));
        palette.setColor(QPalette::WindowText, QColor("#e2e8f0"));
        palette.setColor(QPalette::Base, QColor("#1e293b"));
        palette.setColor(QPalette::AlternateBase, QColor("#334155"));
        palette.setColor(QPalette::Text, QColor("#e2e8f0"));
        palette.setColor(QPalette::Button, QColor("#334155"));
        palette.setColor(QPalette::ButtonText, QColor("#e2e8f0"));
        palette.setColor(QPalette::Highlight, QColor("#3b82f6"));
        palette.setColor(QPalette::HighlightedText, QColor("#ffffff"));
        qApp->setPalette(palette);
        
        qApp->setStyleSheet(
            "QMainWindow { background: #0f172a; }"
            "QStatusBar { background: #1e293b; color: #94a3b8; }"
            "QMenuBar { background: #1e293b; color: #e2e8f0; }"
            "QMenuBar::item:selected { background: #334155; }"
            "QMenu { background: #1e293b; color: #e2e8f0; }"
            "QMenu::item:selected { background: #3b82f6; }"
            "QScrollBar:vertical { background: #1e293b; width: 8px; }"
            "QScrollBar::handle:vertical { background: #334155; border-radius: 4px; }"
        );
    }

    void setupUI() {
        auto* centralWidget = new QWidget();
        auto* mainLayout = new QHBoxLayout(centralWidget);
        mainLayout->setContentsMargins(0, 0, 0, 0);
        mainLayout->setSpacing(0);

        // Sidebar
        auto* sidebar = new QListWidget();
        sidebar->setFixedWidth(200);
        sidebar->setStyleSheet(
            "QListWidget { background: #1e293b; border: none; }"
            "QListWidget::item { padding: 12px; color: #94a3b8; }"
            "QListWidget::item:selected { background: #3b82f6; color: white; }"
            "QListWidget::item:hover { background: #334155; }"
        );
        sidebar->addItem("GPU Performance");
        sidebar->addItem("Inference");
        sidebar->addItem("Settings");
        sidebar->setCurrentRow(0);
        mainLayout->addWidget(sidebar);

        // Content stack
        m_contentStack = new QStackedWidget();
        m_contentStack->addWidget(new GPUPerformancePanel());
        m_contentStack->addWidget(new InferencePanel());
        m_contentStack->addWidget(new QLabel("Settings coming soon..."));
        mainLayout->addWidget(m_contentStack);

        connect(sidebar, &QListWidget::currentRowChanged, 
                m_contentStack, &QStackedWidget::setCurrentIndex);

        setCentralWidget(centralWidget);
    }

    void setupMenus() {
        auto* fileMenu = menuBar()->addMenu("&File");
        fileMenu->addAction("&Open Model...", this, []() {}, QKeySequence::Open);
        fileMenu->addSeparator();
        fileMenu->addAction("E&xit", qApp, &QApplication::quit, QKeySequence::Quit);

        auto* viewMenu = menuBar()->addMenu("&View");
        viewMenu->addAction("GPU &Performance", this, [this]() { 
            m_contentStack->setCurrentIndex(0); 
        });
        viewMenu->addAction("&Inference", this, [this]() { 
            m_contentStack->setCurrentIndex(1); 
        });

        auto* helpMenu = menuBar()->addMenu("&Help");
        helpMenu->addAction("&About", this, [this]() {
            QMessageBox::about(this, "About AI Forge Studio",
                "<h2>AI Forge Studio</h2>"
                "<p>GPU-Accelerated AI Development Environment</p>"
                "<p>Author: M.3R3</p>"
                "<p>Technologies: CUDA, TensorRT, Vulkan, Qt6</p>");
        });
    }

    QStackedWidget* m_contentStack;
};

//=============================================================================
// Main Entry Point
//=============================================================================

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    app.setApplicationName("AI Forge Studio");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("M.3R3");
    
    // High DPI support
    QApplication::setHighDpiScaleFactorRoundingPolicy(
        Qt::HighDpiScaleFactorRoundingPolicy::PassThrough);

    MainWindow window;
    window.show();

    return app.exec();
}

#include "main.moc"
