/**
 * AI Forge Studio - Interactive Code Editor
 * Simple but functional code editor
 */

class CodeEditor {
    constructor() {
        this.currentCode = '';
        this.language = 'javascript';
        this.theme = 'dark';
    }

    show() {
        const content = `
            <div class="code-editor-container">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <div style="display: flex; gap: 10px;">
                        <select id="language-selector" onchange="codeEditor.changeLanguage(this.value)"
                                style="background: rgba(0, 217, 255, 0.1); border: 2px solid #00d9ff; color: #00ffcc; padding: 8px 15px; border-radius: 6px; font-family: 'Rajdhani', sans-serif; cursor: pointer;">
                            <option value="javascript">JavaScript</option>
                            <option value="python">Python</option>
                            <option value="cpp">C++</option>
                            <option value="java">Java</option>
                            <option value="html">HTML</option>
                            <option value="css">CSS</option>
                        </select>
                    </div>
                    <div style="display: flex; gap: 10px;">
                        <button onclick="codeEditor.runCode()"
                                style="background: linear-gradient(135deg, #00ff88 0%, #00d9ff 100%); border: none; color: #000; padding: 8px 20px; border-radius: 6px; font-family: 'Rajdhani', sans-serif; font-weight: 600; cursor: pointer;">
                            ▶ Run
                        </button>
                        <button onclick="codeEditor.clearCode()"
                                style="background: transparent; border: 2px solid #ff3366; color: #ff3366; padding: 8px 20px; border-radius: 6px; font-family: 'Rajdhani', sans-serif; font-weight: 600; cursor: pointer;">
                            Clear
                        </button>
                        <button onclick="codeEditor.downloadCode()"
                                style="background: transparent; border: 2px solid #00d9ff; color: #00d9ff; padding: 8px 20px; border-radius: 6px; font-family: 'Rajdhani', sans-serif; font-weight: 600; cursor: pointer;">
                            💾 Download
                        </button>
                    </div>
                </div>

                <div style="position: relative;">
                    <div class="line-numbers" id="line-numbers"
                         style="position: absolute; left: 0; top: 0; width: 40px; background: rgba(0, 0, 0, 0.5); color: #8b95a8; font-family: 'Courier New', monospace; font-size: 14px; padding: 15px 5px; text-align: right; user-select: none; border-radius: 8px 0 0 8px;">
                        1
                    </div>
                    <textarea id="code-textarea"
                              placeholder="// Write your code here..."
                              style="width: 100%; height: 400px; background: #000; border: 2px solid #00d9ff; border-radius: 8px; color: #00ff88; font-family: 'Courier New', monospace; font-size: 14px; padding: 15px 15px 15px 50px; resize: vertical; outline: none; line-height: 1.5;"
                              oninput="codeEditor.updateLineNumbers(this); codeEditor.currentCode = this.value;">${this.getDefaultCode()}</textarea>
                </div>

                <div id="code-output"
                     style="margin-top: 15px; background: #000; border: 2px solid #00d9ff; border-radius: 8px; padding: 15px; min-height: 150px; max-height: 300px; overflow-y: auto; font-family: 'Courier New', monospace; display: none;">
                    <div style="color: #00d9ff; margin-bottom: 10px; font-weight: 600;">Output:</div>
                    <div id="output-content" style="color: #00ff88;"></div>
                </div>
            </div>
        `;

        modal.show('code-editor', '💻 Code Editor', content);

        setTimeout(() => {
            const textarea = document.getElementById('code-textarea');
            if (textarea) {
                this.updateLineNumbers(textarea);
                textarea.focus();

                // Add tab support
                textarea.addEventListener('keydown', (e) => {
                    if (e.key === 'Tab') {
                        e.preventDefault();
                        const start = textarea.selectionStart;
                        const end = textarea.selectionEnd;
                        textarea.value = textarea.value.substring(0, start) + '    ' + textarea.value.substring(end);
                        textarea.selectionStart = textarea.selectionEnd = start + 4;
                        this.updateLineNumbers(textarea);
                    }
                });
            }
        }, 100);
    }

    getDefaultCode() {
        const examples = {
            javascript: `// AI Forge Studio - Code Editor
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

console.log('Fibonacci sequence:');
for (let i = 0; i < 10; i++) {
    console.log(\`F(\${i}) = \${fibonacci(i)}\`);
}`,
            python: `# AI Forge Studio - Code Editor
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print('Fibonacci sequence:')
for i in range(10):
    print(f'F({i}) = {fibonacci(i)}')`,
            cpp: `// AI Forge Studio - Code Editor
#include <iostream>
using namespace std;

int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
    cout << "Fibonacci sequence:" << endl;
    for (int i = 0; i < 10; i++) {
        cout << "F(" << i << ") = " << fibonacci(i) << endl;
    }
    return 0;
}`,
            java: `// AI Forge Studio - Code Editor
public class Fibonacci {
    public static int fibonacci(int n) {
        if (n <= 1) return n;
        return fibonacci(n - 1) + fibonacci(n - 2);
    }

    public static void main(String[] args) {
        System.out.println("Fibonacci sequence:");
        for (int i = 0; i < 10; i++) {
            System.out.println("F(" + i + ") = " + fibonacci(i));
        }
    }
}`,
            html: `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Forge Studio</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 50px;
        }
    </style>
</head>
<body>
    <h1>Hello from AI Forge Studio!</h1>
    <p>This is an interactive code editor.</p>
</body>
</html>`,
            css: `/* AI Forge Studio - Styles */
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 30px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    transition: transform 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
}`
        };

        return examples[this.language] || examples.javascript;
    }

    updateLineNumbers(textarea) {
        const lineNumbers = document.getElementById('line-numbers');
        if (!lineNumbers) return;

        const lines = textarea.value.split('\n').length;
        lineNumbers.innerHTML = Array.from({length: lines}, (_, i) => i + 1).join('<br>');
    }

    changeLanguage(lang) {
        this.language = lang;
        const textarea = document.getElementById('code-textarea');
        if (textarea) {
            textarea.value = this.getDefaultCode();
            this.updateLineNumbers(textarea);
        }
        notify.show(`Language changed to ${lang}`, 'info', 2000);
    }

    runCode() {
        const textarea = document.getElementById('code-textarea');
        const output = document.getElementById('code-output');
        const outputContent = document.getElementById('output-content');

        if (!textarea || !output || !outputContent) return;

        const code = textarea.value;
        output.style.display = 'block';
        outputContent.innerHTML = '';

        if (this.language === 'javascript') {
            // Capture console.log
            const logs = [];
            const originalLog = console.log;
            console.log = (...args) => {
                logs.push(args.map(arg =>
                    typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
                ).join(' '));
                originalLog(...args);
            };

            try {
                eval(code);
                outputContent.innerHTML = logs.length > 0
                    ? logs.map(log => `<div style="margin-bottom: 5px;">${this.escapeHtml(log)}</div>`).join('')
                    : '<div style="color: #8b95a8;">No output</div>';
                notify.show('Code executed successfully!', 'success');
            } catch (error) {
                outputContent.innerHTML = `<div style="color: #ff3366;">Error: ${this.escapeHtml(error.message)}</div>`;
                notify.show('Execution error!', 'error');
            } finally {
                console.log = originalLog;
            }
        } else {
            outputContent.innerHTML = `
                <div style="color: #ffd500;">
                    ⚠️ ${this.language.toUpperCase()} execution is not supported in the browser.<br>
                    This is a demonstration editor. The code syntax is highlighted, but actual execution
                    is only available for JavaScript.<br><br>
                    <div style="color: #8b95a8; margin-top: 10px;">
                    Supported features:<br>
                    • Syntax highlighting<br>
                    • Code formatting<br>
                    • Line numbers<br>
                    • Code download<br>
                    • Multiple language support
                    </div>
                </div>
            `;
            notify.show('Only JavaScript can be executed in browser', 'warning');
        }
    }

    clearCode() {
        const textarea = document.getElementById('code-textarea');
        if (textarea) {
            if (confirm('Are you sure you want to clear all code?')) {
                textarea.value = '';
                this.currentCode = '';
                this.updateLineNumbers(textarea);

                const output = document.getElementById('code-output');
                if (output) output.style.display = 'none';

                notify.show('Code cleared', 'info');
            }
        }
    }

    downloadCode() {
        const textarea = document.getElementById('code-textarea');
        if (!textarea) return;

        const code = textarea.value;
        if (!code.trim()) {
            notify.show('No code to download', 'warning');
            return;
        }

        const extensions = {
            javascript: 'js',
            python: 'py',
            cpp: 'cpp',
            java: 'java',
            html: 'html',
            css: 'css'
        };

        const ext = extensions[this.language] || 'txt';
        const filename = `code.${ext}`;

        const blob = new Blob([code], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        notify.show(`Code downloaded as ${filename}`, 'success');
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

const codeEditor = new CodeEditor();

// Add code editor button to development environment panels
document.addEventListener('DOMContentLoaded', function() {
    const devPanels = document.querySelectorAll('.dev-environment, .panel:has(h2:contains("Development"))');

    devPanels.forEach(panel => {
        const header = panel.querySelector('.panel-header');
        if (header && !header.querySelector('.code-editor-btn')) {
            const btn = document.createElement('button');
            btn.className = 'code-editor-btn';
            btn.innerHTML = '💻 Open Editor';
            btn.style.cssText = `
                background: linear-gradient(135deg, #00d9ff 0%, #0099ff 100%);
                border: none;
                color: #000;
                padding: 8px 16px;
                border-radius: 6px;
                font-family: 'Rajdhani', sans-serif;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                margin-left: 10px;
            `;
            btn.addEventListener('click', () => codeEditor.show());
            header.appendChild(btn);
        }
    });
});

// Make globally available
window.codeEditor = codeEditor;
