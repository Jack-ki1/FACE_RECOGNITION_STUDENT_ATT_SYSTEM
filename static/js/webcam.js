/**
 * webcam.js
 * ---------
 * Small reusable helper around getUserMedia + canvas capture, used by both
 * the register and attendance pages. No frameworks, just a thin wrapper so
 * each page's inline script stays short.
 */

class FaceCapture {
    constructor(videoElement, canvasElement) {
        this.video = videoElement;
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');
        this.stream = null;
        this.isStreaming = false;
        this.faceDetected = false;
        this.faceDetectionInterval = null;
    }

    async start() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    facingMode: 'user',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }, 
                audio: false 
            });
            this.video.srcObject = this.stream;
            
            this.video.addEventListener('loadedmetadata', () => {
                this.video.play();
                this.isStreaming = true;
                
                // Start face detection monitoring
                this.startFaceDetection();
            });
            
            return true;
        } catch (err) {
            console.error('Error accessing camera:', err);
            // Show user-friendly error message
            if (window.showToast) {
                showToast('Camera access denied. Please enable camera permissions in your browser settings.', 'error');
            }
            return false;
        }
    }

    // Face detection using brightness variance (simple face presence detection)
    startFaceDetection() {
        if (this.faceDetectionInterval) {
            clearInterval(this.faceDetectionInterval);
        }
        
        this.faceDetectionInterval = setInterval(() => {
            if (!this.isStreaming || !this.isVideoPlaying()) return;
            
            // Draw current frame to canvas
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            
            // Get image data
            const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
            const data = imageData.data;
            
            // Calculate brightness variance
            let sum = 0;
            let sumSq = 0;
            let count = 0;
            
            // Sample every 10th pixel for performance
            for (let i = 0; i < data.length; i += 40) {
                const brightness = (data[i] + data[i+1] + data[i+2]) / 3;
                sum += brightness;
                sumSq += brightness * brightness;
                count++;
            }
            
            if (count > 0) {
                const mean = sum / count;
                const variance = (sumSq / count) - (mean * mean);
                
                // If variance is above threshold, assume face is present
                this.faceDetected = variance > 500; // Adjust threshold as needed
                
                // Update UI if available
                if (window.updateFaceDetectionStatus) {
                    updateFaceDetectionStatus(this.faceDetected);
                }
            }
        }, 500); // Check every 500ms
    }
    
    stopFaceDetection() {
        if (this.faceDetectionInterval) {
            clearInterval(this.faceDetectionInterval);
            this.faceDetectionInterval = null;
        }
    }

    capture() {
        const video = this.video;
        const canvas = this.canvas;
        const ctx = this.ctx;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        return canvas.toDataURL('image/jpeg', 0.85);
    }

    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
    }
}

// Utility function to show busy state on buttons
function setBusy(button, text) {
    if (!button) return;
    
    // Store original content
    button.dataset.originalHtml = button.innerHTML;
    
    button.disabled = true;
    button.innerHTML = `<span class="spinner"></span> ${text}`;
    
    // Add loading class for additional styling
    button.classList.add('loading');
}

// Utility function to restore button state
function restoreButton(button) {
    if (!button) return;
    
    button.disabled = false;
    button.innerHTML = button.dataset.originalHtml || button.innerHTML;
    button.classList.remove('loading');
}

// Add utility function to check if an element has a class
function hasClass(element, className) {
    return element.classList.contains(className);
}

// Add utility function to add a class
function addClass(element, className) {
    element.classList.add(className);
}

// Add utility function to remove a class
function removeClass(element, className) {
    element.classList.remove(className);
}

// Add utility function to toggle a class
function toggleClass(element, className) {
    element.classList.toggle(className);
}

// Enhanced utility function to fade in elements
function fadeIn(element, duration = 300) {
    element.style.opacity = 0;
    element.style.display = 'block';
    
    const startTime = Date.now();
    const fadeInInterval = setInterval(() => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        element.style.opacity = progress;
        
        if (progress === 1) {
            clearInterval(fadeInInterval);
        }
    }, 16);
}

// Enhanced utility function to fade out elements
function fadeOut(element, duration = 300) {
    const startTime = Date.now();
    const fadeOutInterval = setInterval(() => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        element.style.opacity = 1 - progress;
        
        if (progress === 1) {
            clearInterval(fadeOutInterval);
            element.style.display = 'none';
        }
    }, 16);
}

// Debounce function for better performance
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

// Throttle function for rate limiting
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}
