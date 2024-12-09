// Error handling wrapper
window.onerror = function(msg, url, lineNo, columnNo, error) {
    console.error('Global error:', {msg, url, lineNo, columnNo, error});
    showError('An unexpected error occurred. Please try again.');
    return false;
};

class CodeExecutor {
    constructor() {
        this.safe_globals = {
            'print': (...args) => { 
                const sanitized = args.map(arg => this.sanitizeOutput(arg));
                this.output.push(sanitized.join(' ')); 
            },
            'range': function*(n) { 
                n = Math.min(Math.max(0, parseInt(n) || 0), 1000000);
                for(let i = 0; i < n; i++) yield i; 
            },
            'len': arr => Array.isArray(arr) ? arr.length : String(arr).length,
            'str': x => this.sanitizeOutput(String(x)),
            'int': x => {
                const parsed = parseInt(x);
                if (isNaN(parsed)) throw new Error('Invalid integer conversion');
                return parsed;
            },
            'float': x => {
                const parsed = parseFloat(x);
                if (isNaN(parsed)) throw new Error('Invalid float conversion');
                return parsed;
            },
            'list': x => Array.from(x).map(item => this.sanitizeOutput(item)),
            'dict': () => ({}),
            'set': x => new Set(Array.from(x).map(item => this.sanitizeOutput(item))),
            'max': (...args) => Math.max(...args),
            'min': (...args) => Math.min(...args),
            'sum': arr => arr.reduce((a, b) => a + b, 0),
            'sleep': ms => new Promise(resolve => setTimeout(resolve, Math.min(ms, 5000))),
            'random': {
                'randint': (min, max) => Math.floor(Math.random() * (max - min + 1)) + min,
                'choice': arr => arr[Math.floor(Math.random() * arr.length)],
                'shuffle': arr => [...arr].sort(() => Math.random() - 0.5)
            },
            'JSON': {
                parse: JSON.parse,
                stringify: JSON.stringify
            }
        };
        this.output = [];
        this.executionCount = 0;
        this.maxExecutions = 2000;
        this.maxCodeLength = 50000;
        this.maxOutputLength = 20000;
        this.executionTimeout = 8000;
    }

    sanitizeOutput(input) {
        if (typeof input === 'string') {
            return input.replace(/[<>]/g, c => ({ '<': '&lt;', '>': '&gt;' })[c]);
        }
        return input;
    }

    async executeCode(code, iterations = 1) {
        this.output = [];
        let result;

        try {
            for (let i = 0; i < iterations; i++) {
                result = await this.executeCodeSafely(code);
                if (result) {
                    this.output.push(`Iteration ${i + 1}: ${result}`);
                }
            }
            return this.output.join('\n');
        } catch (error) {
            throw new Error(`Execution failed: ${error.message}`);
        }
    }

    async executeCodeSafely(code) {
        if (typeof code !== 'string' || code.length > this.maxCodeLength) {
            throw new Error('Invalid code input');
        }

        if (this.executionCount > this.maxExecutions) {
            throw new Error('Maximum execution count exceeded');
        }

        this.executionCount++;

        return new Promise((resolve, reject) => {
            const worker = new Worker(workerUrl);
            const timeout = setTimeout(() => {
                worker.terminate();
                reject(new Error('Execution timed out'));
            }, this.executionTimeout);

            worker.onmessage = (e) => {
                clearTimeout(timeout);
                worker.terminate();
                if (e.data.success) {
                    resolve(e.data.result);
                } else {
                    reject(new Error(e.data.error));
                }
            };

            worker.onerror = (error) => {
                clearTimeout(timeout);
                worker.terminate();
                reject(new Error(error.message));
            };

            worker.postMessage(code);
        });
    }
}

// Initialize components after DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const executor = new CodeExecutor();
    const form = document.getElementById('code-form');
    const codeInput = document.getElementById('code');
    const iterationsInput = document.getElementById('iterations');
    const outputElement = document.getElementById('output');
    const messageElement = document.getElementById('message');
    const clearBtn = document.getElementById('clear-btn');
    const copyBtn = document.getElementById('copy-btn');
    const downloadBtn = document.getElementById('download-btn');
    const darkModeToggle = document.getElementById('darkModeToggle');
    let isExecuting = false;

    // Initialize CodeMirror editor
    const editor = CodeMirror.fromTextArea(codeInput, {
        mode: 'javascript',
        theme: 'monokai',
        lineNumbers: true,
        autoCloseBrackets: true,
        matchBrackets: true,
        indentUnit: 4,
        tabSize: 4,
        lineWrapping: true,
        foldGutter: true,
        gutters: ['CodeMirror-linenumbers', 'CodeMirror-foldgutter'],
        extraKeys: {
            'Ctrl-Space': 'autocomplete',
            'Ctrl-/': 'toggleComment',
            'Ctrl-F': 'findPersistent'
        },
        viewportMargin: Infinity,
        workTime: 20,
        workDelay: 300,
        pollInterval: 100
    });

    // Add debounce utility for performance
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Add optimized event handlers
    const debouncedUpdate = debounce(() => {
        localStorage.setItem('code', editor.getValue());
    }, 1000);

    editor.on('change', debouncedUpdate);

    // Restore saved code if exists
    const savedCode = localStorage.getItem('code');
    if (savedCode) {
        editor.setValue(savedCode);
    }

    // Form submission handler
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (isExecuting) return;

        const code = editor.getValue().trim();
        const iterations = parseInt(iterationsInput.value);

        if (!code) {
            showError('Please enter some code');
            return;
        }

        if (iterations < 1 || iterations > 100) {
            showError('Iterations must be between 1 and 100');
            return;
        }

        try {
            isExecuting = true;
            updateUIState(true);
            const result = await executor.executeCode(code, iterations);
            outputElement.textContent = result;
            messageElement.classList.add('d-none');
        } catch (error) {
            showError(error.message);
        } finally {
            isExecuting = false;
            updateUIState(false);
        }
    });

    // Clear button handler
    clearBtn.addEventListener('click', () => {
        if (isExecuting) return;
        editor.setValue('');
        outputElement.textContent = '';
        messageElement.classList.add('d-none');
        editor.focus();
    });

    // Copy button handler
    copyBtn.addEventListener('click', async () => {
        try {
            const textToCopy = outputElement.textContent;
            if (!textToCopy) {
                showError('No output to copy');
                return;
            }
            
            await navigator.clipboard.writeText(textToCopy);
            showSuccess('Copied to clipboard!');
        } catch (err) {
            showError('Failed to copy to clipboard');
        }
    });

    // Download button handler
    downloadBtn.addEventListener('click', () => {
        const output = outputElement.textContent;
        if (!output) {
            showError('No output to download');
            return;
        }

        const blob = new Blob([output], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `code-output-${new Date().toISOString()}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    // Dark mode toggle handler
    if (localStorage.getItem('darkMode') === 'enabled') {
        document.body.classList.add('dark-mode');
        darkModeToggle.checked = true;
        editor.setOption('theme', 'monokai');
    }

    darkModeToggle.addEventListener('change', () => {
        if (darkModeToggle.checked) {
            document.body.classList.add('dark-mode');
            localStorage.setItem('darkMode', 'enabled');
            editor.setOption('theme', 'monokai');
        } else {
            document.body.classList.remove('dark-mode');
            localStorage.setItem('darkMode', 'disabled');
            editor.setOption('theme', 'default');
        }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            if (e.key === 'Enter' && !isExecuting) {
                e.preventDefault();
                form.dispatchEvent(new Event('submit'));
            } else if (e.key === 'l') {
                e.preventDefault();
                clearBtn.click();
            }
        }
    });

    // Helper functions
    function showError(message) {
        messageElement.textContent = message;
        messageElement.classList.remove('d-none', 'alert-success');
        messageElement.classList.add('alert-danger');
        messageElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function showSuccess(message) {
        messageElement.textContent = message;
        messageElement.classList.remove('d-none', 'alert-danger');
        messageElement.classList.add('alert-success');
        setTimeout(() => messageElement.classList.add('d-none'), 2000);
    }

    function updateUIState(executing) {
        document.body.style.cursor = executing ? 'wait' : 'default';
        form.querySelector('button[type="submit"]').disabled = executing;
        clearBtn.disabled = executing;
        editor.setOption('readOnly', executing);
        iterationsInput.disabled = executing;
    }

    // Prevent accidental navigation
    window.addEventListener('beforeunload', (e) => {
        if (editor.getValue().trim()) {
            e.preventDefault();
            e.returnValue = '';
        }
    });

    // Clean up worker URL when page unloads
    window.addEventListener('unload', () => {
        URL.revokeObjectURL(workerUrl);
    });
});

// Service Worker Registration
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js').catch(error => {
            console.error('ServiceWorker registration failed:', error);
        });
    });
} 