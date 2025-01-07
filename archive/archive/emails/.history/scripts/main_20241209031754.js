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
        this.activeWorkers = new Set();
    }

    cleanup() {
        // Terminate all active workers
        for (const worker of this.activeWorkers) {
            worker.terminate();
        }
        this.activeWorkers.clear();
        this.output = [];
        this.executionCount = 0;
    }

    sanitizeOutput(input) {
        if (input === null || input === undefined) {
            return '';
        }
        if (typeof input === 'string') {
            return input.replace(/[<>]/g, c => ({ '<': '&lt;', '>': '&gt;' })[c]);
        }
        if (typeof input === 'object') {
            try {
                return JSON.stringify(input);
            } catch (e) {
                return String(input);
            }
        }
        return String(input);
    }

    async executeCode(code, iterations = 1) {
        if (!code || typeof code !== 'string') {
            throw new Error('Invalid code input');
        }

        if (iterations < 1 || iterations > 100) {
            throw new Error('Invalid number of iterations');
        }

        try {
            this.cleanup(); // Clean up before starting new execution
            this.output = [];
            
            for (let i = 0; i < iterations; i++) {
                if (this.executionCount >= this.maxExecutions) {
                    throw new Error('Maximum execution count exceeded');
                }
                
                const result = await this.executeCodeSafely(code);
                if (result) {
                    const sanitizedResult = this.sanitizeOutput(result);
                    if (this.output.join('\n').length + sanitizedResult.length > this.maxOutputLength) {
                        throw new Error('Maximum output length exceeded');
                    }
                    this.output.push(`Iteration ${i + 1}: ${sanitizedResult}`);
                }
                this.executionCount++;
            }
            
            return this.output.join('\n');
        } catch (error) {
            this.cleanup(); // Clean up on error
            throw new Error(`Execution failed: ${error.message}`);
        }
    }

    async executeCodeSafely(code) {
        if (code.length > this.maxCodeLength) {
            throw new Error('Code exceeds maximum length');
        }

        return new Promise((resolve, reject) => {
            try {
                const workerCode = `
                    self.onmessage = function(e) {
                        try {
                            const code = e.data;
                            const result = new Function('return ' + code)();
                            self.postMessage({ success: true, result: result });
                        } catch (error) {
                            self.postMessage({ success: false, error: error.message });
                        }
                    };
                `;
                
                const blob = new Blob([workerCode], { type: 'application/javascript' });
                const workerUrl = URL.createObjectURL(blob);
                const worker = new Worker(workerUrl);
                
                this.activeWorkers.add(worker);
                
                const timeout = setTimeout(() => {
                    this.activeWorkers.delete(worker);
                    worker.terminate();
                    URL.revokeObjectURL(workerUrl);
                    reject(new Error('Execution timed out'));
                }, this.executionTimeout);

                worker.onmessage = (e) => {
                    clearTimeout(timeout);
                    this.activeWorkers.delete(worker);
                    worker.terminate();
                    URL.revokeObjectURL(workerUrl);
                    
                    if (e.data.success) {
                        resolve(e.data.result);
                    } else {
                        reject(new Error(e.data.error));
                    }
                };

                worker.onerror = (error) => {
                    clearTimeout(timeout);
                    this.activeWorkers.delete(worker);
                    worker.terminate();
                    URL.revokeObjectURL(workerUrl);
                    reject(new Error(error.message));
                };

                worker.postMessage(code);
            } catch (error) {
                reject(new Error(`Worker initialization failed: ${error.message}`));
            }
        });
    }
}

class NativeAppFeatures {
    constructor() {
        this.isExecuting = false;
        this.editor = null;
        this.executor = new CodeExecutor();
        
        // Initialize all features
        this.setupServiceWorker();
        this.setupDragAndDrop();
        this.setupKeyboardShortcuts();
        this.setupNotifications();
        this.setupFileSystemAccess();
        this.setupCodeMirror();
        this.setupEventListeners();
    }

    setupCodeMirror() {
        const codeInput = document.getElementById('code');
        this.editor = CodeMirror.fromTextArea(codeInput, {
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

        // Restore saved code if exists
        const savedCode = localStorage.getItem('code');
        if (savedCode) {
            this.editor.setValue(savedCode);
        }

        // Auto-save code changes
        this.editor.on('change', this.debounce(() => {
            localStorage.setItem('code', this.editor.getValue());
        }, 1000));
    }

    setupEventListeners() {
        // Form submission
        const form = document.getElementById('code-form');
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.executeCode();
        });

        // Clear button
        const clearBtn = document.getElementById('clear-btn');
        clearBtn.addEventListener('click', () => {
            if (this.isExecuting) return;
            this.editor.setValue('');
            document.getElementById('output').textContent = '';
            this.showNotification('Editor cleared', 'info');
        });

        // Copy button
        const copyBtn = document.getElementById('copy-btn');
        copyBtn.addEventListener('click', async () => {
            const output = document.getElementById('output').textContent;
            if (!output) {
                this.showNotification('No output to copy', 'warning');
                return;
            }
            
            try {
                await navigator.clipboard.writeText(output);
                this.showNotification('Copied to clipboard!', 'success');
            } catch (err) {
                this.showNotification('Failed to copy to clipboard', 'danger');
            }
        });

        // Download button
        const downloadBtn = document.getElementById('download-btn');
        downloadBtn.addEventListener('click', () => {
            const output = document.getElementById('output').textContent;
            if (!output) {
                this.showNotification('No output to download', 'warning');
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
            this.showNotification('Download started!', 'success');
        });

        // Theme toggle
        const darkModeToggle = document.getElementById('theme-toggle');
        if (localStorage.getItem('darkMode') === 'enabled') {
            document.body.classList.add('dark-mode');
            this.editor.setOption('theme', 'monokai');
        }

        darkModeToggle.addEventListener('click', () => {
            document.body.classList.toggle('dark-mode');
            const isDarkMode = document.body.classList.contains('dark-mode');
            localStorage.setItem('darkMode', isDarkMode ? 'enabled' : 'disabled');
            this.editor.setOption('theme', isDarkMode ? 'monokai' : 'default');
            darkModeToggle.innerHTML = isDarkMode 
                ? '<i class="bi bi-sun-fill" aria-hidden="true"></i>' 
                : '<i class="bi bi-moon-fill" aria-hidden="true"></i>';
        });

        // Prevent accidental navigation
        window.addEventListener('beforeunload', (e) => {
            if (this.editor.getValue().trim()) {
                e.preventDefault();
                e.returnValue = '';
            }
        });
    }

    async executeCode() {
        if (this.isExecuting) {
            this.showNotification('Already executing code', 'warning');
            return;
        }

        const code = this.editor.getValue().trim();
        const iterations = parseInt(document.getElementById('iterations').value);

        if (!code) {
            this.showNotification('Please enter some code', 'warning');
            return;
        }

        if (iterations < 1 || iterations > 100) {
            this.showNotification('Iterations must be between 1 and 100', 'warning');
            return;
        }

        try {
            this.isExecuting = true;
            this.updateUIState(true);
            const result = await this.executor.executeCode(code, iterations);
            document.getElementById('output').textContent = result;
            this.showNotification('Code executed successfully!', 'success');
        } catch (error) {
            this.showNotification(error.message, 'danger');
        } finally {
            this.isExecuting = false;
            this.updateUIState(false);
        }
    }

    updateUIState(isLoading) {
        const startBtn = document.getElementById('start-btn');
        const elements = [
            this.editor,
            document.getElementById('iterations'),
            document.getElementById('clear-btn'),
            document.getElementById('copy-btn'),
            document.getElementById('download-btn'),
            startBtn
        ];
        
        elements.forEach(el => {
            if (el) el.disabled = isLoading;
        });

        if (startBtn) {
            startBtn.innerHTML = isLoading 
                ? '<i class="bi bi-hourglass-split"></i> Running...'
                : '<i class="bi bi-play-fill"></i> Execute';
        }

        document.body.style.cursor = isLoading ? 'wait' : 'default';
    }

    // Other methods remain unchanged...
}

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    new NativeAppFeatures();
});

// Service Worker Registration
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/service-worker.js').catch(error => {
            console.error('ServiceWorker registration failed:', error);
        });
    });
}