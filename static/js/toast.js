// Enhanced Toast notification system
let toastStack = document.getElementById('toast-stack');
let toasts = []; // Track active toasts

// Process any flash messages from the server
document.addEventListener('DOMContentLoaded', function() {
    const flashData = document.getElementById('flash-data');
    if (flashData) {
        try {
            const messages = JSON.parse(flashData.textContent);
            messages.forEach(msg => {
                if (msg.text) {
                    showToast(msg.text, msg.category || 'info');
                }
            });
        } catch (e) {
            console.error('Failed to parse flash data:', e);
        }
    }
});

function showToast(message, type = 'info', duration = 5000) {
    // Create toast container if it doesn't exist
    if (!toastStack) {
        toastStack = document.createElement('div');
        toastStack.id = 'toast-stack';
        toastStack.style.cssText = 
            'position: fixed;' +
            'top: 24px;' +
            'right: 24px;' +
            'z-index: 1000;' +
            'display: flex;' +
            'flex-direction: column;' +
            'gap: 8px;' +
            'max-width: 400px;';
        document.body.appendChild(toastStack);
    }
    
    const toast = document.createElement('div');
    toast.className = \`toast ${type}\`;
    
    // Map category names to appropriate icons and colors
    let icon = 'info';
    let bgColor = 'var(--brand-primary)';
    
    switch(type.toLowerCase()) {
        case 'success':
        case 'ok':
            icon = 'check-circle';
            bgColor = 'var(--success)';
            break;
        case 'error':
        case 'danger':
            icon = 'alert-circle';
            bgColor = 'var(--error)';
            break;
        case 'warning':
        case 'warn':
            icon = 'alert-triangle';
            bgColor = 'var(--warning)';
            break;
        case 'info':
        default:
            icon = 'info';
            bgColor = 'var(--brand-primary)';
            break;
    }
    
    toast.innerHTML = \`
        <i data-lucide="\${icon}" style="color: white; width: 20px; height: 20px;"></i>
        <span style="flex: 1; font-weight: 500;">\${message}</span>
        <button type="button" class="toast-close" style="
            background: none; 
            border: none; 
            color: white; 
            cursor: pointer; 
            padding: 4px; 
            border-radius: 4px;
            opacity: 0.8;
            transition: opacity 0.2s;
        " onclick="this.closest('.toast').remove();">&times;</button>
    \`;
    
    // Add toast to the stack
    toastStack.prepend(toast);
    
    // Store reference to this toast
    const toastId = Date.now() + Math.random();
    toast.dataset.toastId = toastId;
    toasts.push(toastId);
    
    // Re-render icons after adding new toast
    if (window.lucide) {
        lucide.createIcons();
    }
    
    // Add entrance animation
    setTimeout(() => {
        toast.style.transform = 'translateX(0)';
        toast.style.opacity = '1';
    }, 10);
    
    // Auto-remove after specified duration
    if (duration > 0) {
        setTimeout(() => {
            removeToast(toast, toastId);
        }, duration);
    }
    
    // Add hover to pause auto-dismiss
    toast.addEventListener('mouseenter', () => {
        toast.dataset.paused = 'true';
    });
    
    toast.addEventListener('mouseleave', () => {
        toast.dataset.paused = 'false';
        if (duration > 0) {
            setTimeout(() => {
                if (toast.dataset.paused !== 'true') {
                    removeToast(toast, toastId);
                }
            }, duration);
        }
    });
    
    // Click to dismiss
    toast.addEventListener('click', (e) => {
        if (!e.target.classList.contains('toast-close')) {
            removeToast(toast, toastId);
        }
    });
    
    return toastId;
}

function removeToast(toast, toastId) {
    // Remove from active toasts array
    toasts = toasts.filter(id => id !== toastId);
    
    // Animate removal
    toast.style.transition = 'all 0.3s ease';
    toast.style.transform = 'translateX(100%)';
    toast.style.opacity = '0';
    
    setTimeout(() => {
        toast.remove();
    }, 300);
}

// Function to clear all toasts
function clearAllToasts() {
    toasts.forEach(toastId => {
        const toast = document.querySelector(\`[data-toast-id="\${toastId}"]\`);
        if (toast) {
            toast.remove();
        }
    });
    toasts = [];
}

// Function to show multiple toasts
function showMultipleToasts(messages) {
    messages.forEach(msg => {
        showToast(msg.text, msg.type || 'info', msg.duration || 5000);
    });
}

// Expose functions globally so they can be used elsewhere
window.showToast = showToast;
window.clearAllToasts = clearAllToasts;
window.showMultipleToasts = showMultipleToasts;
