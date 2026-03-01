// Elements
const imageInput = document.getElementById('imageInput');
const uploadBtn = document.getElementById('uploadBtn');
const mainContent = document.getElementById('mainContent');
const processBtn = document.getElementById('processBtn');
const resultsList = document.getElementById('resultsList');

// Canvases
const originalCanvas = document.getElementById('originalCanvas');
const originalCtx = originalCanvas.getContext('2d', { willReadFrequently: true });
const filterCanvas = document.getElementById('filterCanvas');
const filterCtx = filterCanvas.getContext('2d', { willReadFrequently: true });

// Sliders
const cropTop = document.getElementById('cropTop');
const cropBottom = document.getElementById('cropBottom');
const cropLeft = document.getElementById('cropLeft');
const cropRight = document.getElementById('cropRight');
const sliders = [cropTop, cropBottom, cropLeft, cropRight];

// State
let loadedImage = null;
let imageSegmenter = null;
let currentMask = null; // refined foreground mask from mediapipe (0/255 per pixel)
let mediaPipeReadyPromise = null;

const MP_CONFIDENCE_HIGH = 0.55;
const MP_CONFIDENCE_LOW = 0.35;
const MANUAL_SEGMENTATION_ONLY = false;

// Initialize MediaPipe Tasks
async function initMediaPipe() {
    if (MANUAL_SEGMENTATION_ONLY) {
        return;
    }

    try {
        const vision = await window.TasksVision.FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );

        const baseOptions = {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
        };

        try {
            imageSegmenter = await window.TasksVision.ImageSegmenter.createFromOptions(vision, {
                baseOptions: { ...baseOptions, delegate: "GPU" },
                runningMode: "IMAGE",
                outputCategoryMask: true,
                outputConfidenceMasks: true
            });
        } catch (gpuError) {
            console.warn("GPU segmenter init failed, retrying on CPU.", gpuError);
            imageSegmenter = await window.TasksVision.ImageSegmenter.createFromOptions(vision, {
                baseOptions: { ...baseOptions, delegate: "CPU" },
                runningMode: "IMAGE",
                outputCategoryMask: true,
                outputConfidenceMasks: true
            });
        }
        console.log("MediaPipe Initialized.");
    } catch (e) {
        console.error("Error initializing MediaPipe:", e);
    }
}
mediaPipeReadyPromise = initMediaPipe();

async function runBodySegmentation() {
    if (MANUAL_SEGMENTATION_ONLY || !loadedImage) return;

    if (!imageSegmenter && mediaPipeReadyPromise) {
        await mediaPipeReadyPromise;
    }
    if (!imageSegmenter) return;

    try {
        console.log("Running MediaPipe Segmentation...");
        const segmentationResult = await imageSegmenter.segment(originalCanvas);
        currentMask = buildRefinedMediaPipeMask(segmentationResult, originalCanvas.width, originalCanvas.height);
    } catch (e) {
        console.error("Segmentation failed:", e);
        currentMask = null;
    }
}

// Event Listeners
uploadBtn.addEventListener('click', () => imageInput.click());

imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        const img = new Image();
        img.onload = async () => {
            loadedImage = img;

            // Set canvas dimensions based on image, maintaining aspect ratio up to 800px width
            const maxWidth = 800;
            const targetWidth = Math.min(img.width, maxWidth);
            const scale = targetWidth / img.width;
            const targetHeight = img.height * scale;

            [originalCanvas, filterCanvas].forEach(canvas => {
                canvas.width = targetWidth;
                canvas.height = targetHeight;
            });

            mainContent.style.display = 'grid'; // Show the UI

            // Draw initial image to get ImageData for MediaPipe
            originalCtx.drawImage(img, 0, 0, targetWidth, targetHeight);

            // Run segmentation after upload; waits for model initialization if needed.
            await runBodySegmentation();

            drawOriginalImage();
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
});

sliders.forEach(slider => {
    slider.addEventListener('input', drawOriginalImage);
});

processBtn.addEventListener('click', async () => {
    if (!loadedImage) return;

    // 1. Get the cropped image data masked by Advanced Segmentation
    const bounds = getBounds();

    if (bounds.w <= 0 || bounds.h <= 0) {
        alert("Invalid boundaries selected.");
        return;
    }

    const baseImageData = getBaseImageData();
    const fullImageData = cloneImageData(baseImageData);

    if (!MANUAL_SEGMENTATION_ONLY) {
        if (!currentMask) {
            await runBodySegmentation();
        }

        const finalMaskData = generateAdvancedSkinMask(fullImageData, currentMask, bounds);
        for (let i = 0; i < finalMaskData.length; i++) {
            if (finalMaskData[i] === 0) {
                fullImageData.data[i * 4] = 0;
                fullImageData.data[i * 4 + 1] = 0;
                fullImageData.data[i * 4 + 2] = 0;
                fullImageData.data[i * 4 + 3] = 255;
            }
        }
    }

    // Now extract specifically the cropped boundary
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = bounds.w;
    tempCanvas.height = bounds.h;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.putImageData(MANUAL_SEGMENTATION_ONLY ? baseImageData : fullImageData, -bounds.x, -bounds.y);

    const croppedImageData = tempCtx.getImageData(0, 0, bounds.w, bounds.h);

    // 2. Apply Filters
    const octaStep = applyStepDownExponential(croppedImageData);
    const cannyStep = applyCannySobel(octaStep);

    // Write Canny output to the filter canvas
    filterCtx.fillStyle = '#000';
    filterCtx.fillRect(0, 0, filterCanvas.width, filterCanvas.height);
    filterCtx.putImageData(cannyStep, bounds.x, bounds.y);

    // 3. Find and Score Lesions
    // Find *all* components
    const allComponents = detectRawComponents(cannyStep, bounds.x, bounds.y);

    // Validate them as lesions (circles/ellipses)
    const validLesions = filterValidShapes(allComponents);

    // Score the valid ones
    const scoredLesions = filterAndScoreLesions(validLesions, baseImageData, bounds);

    // 4. Detection-first overlay (boxes + circles), then show scoring details in results panel
    drawDetectionOverlay(validLesions);
    displayResults(scoredLesions);
});

// Helper functions for UI
function getBounds() {
    const t = parseInt(cropTop.value) / 100;
    const b = parseInt(cropBottom.value) / 100;
    const l = parseInt(cropLeft.value) / 100;
    const r = parseInt(cropRight.value) / 100;

    return {
        x: Math.round(originalCanvas.width * l),
        y: Math.round(originalCanvas.height * t),
        w: Math.round(originalCanvas.width * (r - l)),
        h: Math.round(originalCanvas.height * (b - t))
    };
}

function getBaseImageData() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = originalCanvas.width;
    tempCanvas.height = originalCanvas.height;
    const tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true });
    tempCtx.drawImage(loadedImage, 0, 0, tempCanvas.width, tempCanvas.height);
    return tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
}

function cloneImageData(imageData) {
    const copy = new ImageData(imageData.width, imageData.height);
    copy.data.set(imageData.data);
    return copy;
}

function rgbToHsv(r, g, b) {
    r /= 255, g /= 255, b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h, s, v = max;
    const d = max - min;
    s = max === 0 ? 0 : d / max;
    if (max === min) {
        h = 0;
    } else {
        switch (max) {
            case r: h = (g - b) / d + (g < b ? 6 : 0); break;
            case g: h = (b - r) / d + 2; break;
            case b: h = (r - g) / d + 4; break;
        }
        h /= 6;
    }
    return [Math.round(h * 180), Math.round(s * 255), Math.round(v * 255)]; // OpenCV scaling
}

function rgbToYcrcb(r, g, b) {
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    const cr = (r - y) * 0.713 + 128;
    const cb = (b - y) * 0.564 + 128;
    return [y, cr, cb];
}

function buildRefinedMediaPipeMask(segmentationResult, width, height) {
    const totalPixels = width * height;
    const refinedMask = new Uint8Array(totalPixels);

    if (!segmentationResult || !segmentationResult.categoryMask) {
        return refinedMask;
    }

    const categoryData = segmentationResult.categoryMask.getAsFloat32Array();
    const confidenceMasks = segmentationResult.confidenceMasks;
    const foregroundConfidenceMask = confidenceMasks?.length > 1
        ? confidenceMasks[1]
        : confidenceMasks?.[0];
    const confidenceData = foregroundConfidenceMask
        ? foregroundConfidenceMask.getAsFloat32Array()
        : null;
    const usableLength = Math.min(totalPixels, categoryData.length, confidenceData ? confidenceData.length : totalPixels);

    for (let i = 0; i < usableLength; i++) {
        const isForegroundCategory = categoryData[i] !== 0;
        const confidence = confidenceData ? confidenceData[i] : (isForegroundCategory ? 1 : 0);

        if (confidence >= MP_CONFIDENCE_HIGH || (confidence >= MP_CONFIDENCE_LOW && isForegroundCategory)) {
            refinedMask[i] = 255;
        }
    }

    // Close small holes, then open tiny speckles before keeping the largest body component.
    let cleanedMask = applyDilation(refinedMask, width, height, 2);
    cleanedMask = applyErosion(cleanedMask, width, height, 2);
    cleanedMask = applyErosion(cleanedMask, width, height, 1);
    cleanedMask = applyDilation(cleanedMask, width, height, 1);
    return keepLargestComponent(cleanedMask, width, height);
}

// === ADVANCED SEGMENTATION PIPELINE ===
function generateAdvancedSkinMask(imageData, mediapipeMask, cropBounds) {
    const { width, height, data } = imageData;
    const totalPixels = width * height;
    let skinMask = new Uint8Array(totalPixels);

    // Step 1: Color-based Skin Mask (HSV + YCrCb)
    for (let i = 0; i < totalPixels; i++) {
        const r = data[i * 4];
        const g = data[i * 4 + 1];
        const b = data[i * 4 + 2];

        const [h, s, v] = rgbToHsv(r, g, b);
        const [yY, cr, cb] = rgbToYcrcb(r, g, b);

        // Typical skin thresholds (approximated for JS)
        const isHSV = (h >= 0 && h <= 20) && (s >= 40 && s <= 255) && (v >= 60 && v <= 255);
        const isYCrCb = (cr >= 135 && cr <= 180) && (cb >= 85 && cb <= 135) && (yY >= 50);

        if (isHSV && isYCrCb) {
            skinMask[i] = 255;
        }
    }

    // Step 2: Morphological Cleanup (Simplified Open then Close with box/ellipse logic)
    // For JS execution speed, we'll do a basic pass of erosion then dilation.
    skinMask = applyErosion(skinMask, width, height, 2);
    skinMask = applyDilation(skinMask, width, height, 4); // Dilation > Erosion = net close

    // Step 3: MediaPipe Fusion
    if (mediapipeMask) {
        const expandedMpMask = applyDilation(mediapipeMask, width, height, 2);
        let fusedArea = 0;
        const fusedMask = new Uint8Array(totalPixels);

        for (let i = 0; i < totalPixels; i++) {
            if (skinMask[i] === 255 && expandedMpMask[i] === 255) { // logical AND + tiny MP expansion
                fusedMask[i] = 255;
                fusedArea++;
            }
        }

        // Use fused mask when robust; otherwise trust cleaned MP mask directly.
        if (fusedArea > totalPixels * 0.03) {
            skinMask = fusedMask;
        } else {
            skinMask = expandedMpMask;
        }
    }

    // Step 4: Keep main connected body component
    skinMask = keepLargestComponent(skinMask, width, height);

    // Step 5: Apply Exclusions (Crop boundaries)
    // We treat the area outside the user's sliders as "neck/table" exclusion zones
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (x < cropBounds.x || x > (cropBounds.x + cropBounds.w) ||
                y < cropBounds.y || y > (cropBounds.y + cropBounds.h)) {
                skinMask[y * width + x] = 0;
            }
        }
    }

    // Step 6: Keep largest one last time after cutting
    skinMask = keepLargestComponent(skinMask, width, height);

    return skinMask;
}

function buildCircularOffsets(radius) {
    const offsets = [];
    const r2 = radius * radius;
    for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
            if ((dx * dx + dy * dy) <= r2) {
                offsets.push([dx, dy]);
            }
        }
    }
    return offsets;
}

// Morph helper
function applyErosion(mask, w, h, radius) {
    if (radius <= 0) return mask.slice();
    const out = new Uint8Array(w * h);
    const offsets = buildCircularOffsets(radius);

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const idx = y * w + x;
            if (mask[idx] !== 255) continue;

            let keep = true;
            for (const [dx, dy] of offsets) {
                const nx = x + dx;
                const ny = y + dy;
                if (nx < 0 || nx >= w || ny < 0 || ny >= h || mask[ny * w + nx] === 0) {
                    keep = false;
                    break;
                }
            }

            if (keep) {
                out[idx] = 255;
            }
        }
    }
    return out;
}

function applyDilation(mask, w, h, radius) {
    if (radius <= 0) return mask.slice();
    const out = new Uint8Array(w * h);
    const offsets = buildCircularOffsets(radius);

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            if (mask[y * w + x] !== 255) continue;

            for (const [dx, dy] of offsets) {
                const nx = x + dx;
                const ny = y + dy;
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    out[ny * w + nx] = 255;
                }
            }
        }
    }
    return out;
}

function keepLargestComponent(mask, w, h) {
    const total = w * h;
    const visited = new Uint8Array(total);
    const queue = new Int32Array(total);
    let largestSize = 0;
    let largestPixels = null;

    // Find all components in the mask.
    for (let idx = 0; idx < total; idx++) {
        if (mask[idx] !== 255 || visited[idx]) continue;

        let head = 0;
        let tail = 0;
        const componentPixels = [];

        visited[idx] = 1;
        queue[tail++] = idx;

        while (head < tail) {
            const current = queue[head++];
            componentPixels.push(current);
            const x = current % w;
            const y = Math.floor(current / w);

            if (x > 0) {
                const left = current - 1;
                if (!visited[left] && mask[left] === 255) {
                    visited[left] = 1;
                    queue[tail++] = left;
                }
            }
            if (x < w - 1) {
                const right = current + 1;
                if (!visited[right] && mask[right] === 255) {
                    visited[right] = 1;
                    queue[tail++] = right;
                }
            }
            if (y > 0) {
                const up = current - w;
                if (!visited[up] && mask[up] === 255) {
                    visited[up] = 1;
                    queue[tail++] = up;
                }
            }
            if (y < h - 1) {
                const down = current + w;
                if (!visited[down] && mask[down] === 255) {
                    visited[down] = 1;
                    queue[tail++] = down;
                }
            }
        }

        if (componentPixels.length > largestSize) {
            largestSize = componentPixels.length;
            largestPixels = componentPixels;
        }
    }

    // Reconstruct mask with ONLY the largest component
    const finalMask = new Uint8Array(total);
    if (!largestPixels) return finalMask;

    for (const idx of largestPixels) {
        finalMask[idx] = 255;
    }

    return finalMask;
}

function drawOriginalImage() {
    if (!loadedImage) return;

    // Clear
    originalCtx.clearRect(0, 0, originalCanvas.width, originalCanvas.height);

    // Draw base image
    originalCtx.drawImage(loadedImage, 0, 0, originalCanvas.width, originalCanvas.height);

    // Overlay segmentation result by tinting detected background.
    if (!MANUAL_SEGMENTATION_ONLY && currentMask) {
        const imgData = originalCtx.getImageData(0, 0, originalCanvas.width, originalCanvas.height);
        for (let i = 0; i < currentMask.length; i++) {
            if (currentMask[i] === 0) {
                imgData.data[i * 4] = Math.min(255, imgData.data[i * 4] + 80);
                imgData.data[i * 4 + 1] = Math.max(0, imgData.data[i * 4 + 1] - 40);
                imgData.data[i * 4 + 2] = Math.max(0, imgData.data[i * 4 + 2] - 40);
            }
        }
        originalCtx.putImageData(imgData, 0, 0);
    }

    // Draw boundary overlay
    const bounds = getBounds();
    originalCtx.fillStyle = 'rgba(0, 0, 0, 0.6)';

    // Top
    originalCtx.fillRect(0, 0, originalCanvas.width, bounds.y);
    // Bottom
    originalCtx.fillRect(0, bounds.y + bounds.h, originalCanvas.width, originalCanvas.height - (bounds.y + bounds.h));
    // Left
    originalCtx.fillRect(0, bounds.y, bounds.x, bounds.h);
    // Right
    originalCtx.fillRect(bounds.x + bounds.w, bounds.y, originalCanvas.width - (bounds.x + bounds.w), bounds.h);

    // Draw border box
    originalCtx.strokeStyle = '#3b82f6';
    originalCtx.lineWidth = 2;
    originalCtx.strokeRect(bounds.x, bounds.y, bounds.w, bounds.h);
}

// core image processing functions
const applyStepDownExponential = (imageData) => {
    const { width, height, data } = imageData;
    const outData = new ImageData(width, height);
    const decay = 0.85; // This coefficient controls the subtraction strength

    for (let y = 1; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const pIdx = ((y - 1) * width + x) * 4; // Reference to the "superficial" pixel above

            // We subtract a portion of the overlying signal from the current pixel
            outData.data[idx] = Math.max(0, data[idx] - data[pIdx] * (1 - decay));
            outData.data[idx + 1] = Math.max(0, data[idx + 1] - data[pIdx] * (1 - decay));
            outData.data[idx + 2] = Math.max(0, data[idx + 2] - data[pIdx] * (1 - decay));
            outData.data[idx + 3] = 255;
        }
    }
    return outData;
};

const applyCannySobel = (imageData) => {
    const { width, height, data } = imageData;
    const outData = new ImageData(width, height);

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = (y * width + x) * 4;

            // Sobel operator kernels to find intensity changes
            const gx = (-1 * data[((y - 1) * width + (x - 1)) * 4]) + (1 * data[((y - 1) * width + (x + 1)) * 4]) +
                (-2 * data[(y * width + (x - 1)) * 4]) + (2 * data[(y * width + (x + 1)) * 4]) +
                (-1 * data[((y + 1) * width + (x - 1)) * 4]) + (1 * data[((y + 1) * width + (x + 1)) * 4]);

            const gy = (-1 * data[((y - 1) * width + (x - 1)) * 4]) + (-2 * data[((y - 1) * width + x) * 4]) + (-1 * data[((y - 1) * width + (x + 1)) * 4]) +
                (1 * data[((y + 1) * width + (x - 1)) * 4]) + (2 * data[((y + 1) * width + x) * 4]) + (1 * data[((y + 1) * width + (x + 1)) * 4]);

            // Calculate total gradient magnitude
            const mag = Math.sqrt(gx * gx + gy * gy);

            // Thresholding: Only keep strong boundaries
            const val = mag > 55 ? 255 : 0;
            outData.data[idx] = outData.data[idx + 1] = outData.data[idx + 2] = val;
            outData.data[idx + 3] = 255;
        }
    }
    return outData;
};

// Lesion Detection Logic
function detectRawComponents(cannyImageData, offsetX, offsetY) {
    const { width, height, data } = cannyImageData;
    const visited = new Uint8Array(width * height);
    const components = [];

    // Flood fill / Connected Components to find isolated boundaries
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            if (data[idx * 4] === 255 && !visited[idx]) {
                const componentBox = extractBlob(x, y, width, height, data, visited);
                if (componentBox.area > 15) { // Minimum threshold to prevent single-pixel noise
                    componentBox.offsetX = offsetX;
                    componentBox.offsetY = offsetY;
                    components.push(componentBox);
                }
            }
        }
    }
    return components;
}

function filterValidShapes(components) {
    const valid = [];
    components.forEach(comp => {
        // Discard huge areas that are likely boundaries of the image or massive artifacts
        if (comp.w * comp.h < 50000) {
            // Check if shape is roughly circular/elliptical (not just a super long straight line)
            const aspect = Math.max(comp.w, comp.h) / Math.min(comp.w, comp.h);
            if (aspect < 4.5 && comp.w >= 3 && comp.h >= 3) {
                valid.push(comp);
            }
        }
    });
    return valid;
}

function extractBlob(startX, startY, width, height, data, visited) {
    const queue = [[startX, startY]];
    let minX = startX, maxX = startX, minY = startY, maxY = startY;
    let area = 0;

    // Stack for pixels belonging to this blob hull
    const pixels = [];

    visited[startY * width + startX] = 1;

    while (queue.length > 0) {
        const [x, y] = queue.shift();
        pixels.push([x, y]);
        area++;

        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);

        // 8-way connectivity
        const neighbors = [
            [x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1],
            [x - 1, y - 1], [x + 1, y - 1], [x - 1, y + 1], [x + 1, y + 1]
        ];

        for (const [nx, ny] of neighbors) {
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                const idx = ny * width + nx;
                if (!visited[idx] && data[idx * 4] === 255) {
                    visited[idx] = 1;
                    queue.push([nx, ny]);
                }
            }
        }
    }

    return {
        x: minX,
        y: minY,
        w: maxX - minX + 1,
        h: maxY - minY + 1,
        area: area,
        pixels: pixels
    };
}

function filterAndScoreLesions(lesions, originalImage, cropBounds) {
    const scoredLesions = [];

    lesions.forEach((lesion, index) => {
        // Calculate points based on ABCD (Asymmetry, Border, Color, Diameter/Irregularity)
        let score = 0;
        let reasons = [];

        // 1. Asymmetry (Center of mass vs bounding box center)
        let sumX = 0, sumY = 0;
        lesion.pixels.forEach(p => { sumX += p[0]; sumY += p[1]; });
        const cx = sumX / lesion.area;
        const cy = sumY / lesion.area;
        const boxCx = lesion.x + lesion.w / 2;
        const boxCy = lesion.y + lesion.h / 2;
        const asymmetry = Math.sqrt(Math.pow(cx - boxCx, 2) + Math.pow(cy - boxCy, 2));
        const maxExpectedAsymmetry = Math.min(lesion.w, lesion.h) * 0.15; // 15% tolerance

        if (asymmetry > maxExpectedAsymmetry) {
            score++;
            reasons.push('Asymmetry (+1)');
        }

        // 2. Irregular Borders (Perimeter vs Area compactness)
        // Here, the lesion "area" is just the boundary pixels. So standard compactness isn't 1:1.
        // We evaluate filling. If the real bounding area vs boundary pixels is high, it's complex.
        const boxArea = lesion.w * lesion.h;
        const fillRatio = lesion.area / boxArea;
        // A perfect circle boundary will have a low fill ratio but a consistent one.
        // If fill ratio is too high/low, border is irregular.
        if (fillRatio < 0.1 || fillRatio > 0.4) {
            score++;
            reasons.push('Irregular Borders (+1)');
        }

        // 3. More than 2 colors
        // Extract inner pixels from the original image based on bounding box (a simplification)
        let rVals = [], gVals = [], bVals = [], intVals = [];

        // Safely map bounding box back to original image coordinates
        const globalX = lesion.offsetX + lesion.x;
        const globalY = lesion.offsetY + lesion.y;

        for (let innerY = globalY; innerY < globalY + lesion.h; innerY++) {
            for (let innerX = globalX; innerX < globalX + lesion.w; innerX++) {
                if (innerX >= 0 && innerX < originalImage.width && innerY >= 0 && innerY < originalImage.height) {
                    const i = (innerY * originalImage.width + innerX) * 4;
                    rVals.push(originalImage.data[i]);
                    gVals.push(originalImage.data[i + 1]);
                    bVals.push(originalImage.data[i + 2]);

                    // Grayscale intensity for internal irregularity
                    const gray = 0.299 * originalImage.data[i] + 0.587 * originalImage.data[i + 1] + 0.114 * originalImage.data[i + 2];
                    intVals.push(gray);
                }
            }
        }

        // Simple color quantization (split RGB space)
        let distinctColors = new Set();
        for (let i = 0; i < rVals.length; i += Math.floor(rVals.length / 100) + 1) { // Sample ~100 points max to save time
            // Quantize to 4 levels per channel (64 possible colors)
            const rq = Math.floor(rVals[i] / 64);
            const gq = Math.floor(gVals[i] / 64);
            const bq = Math.floor(bVals[i] / 64);
            distinctColors.add(`${rq}-${gq}-${bq}`);
        }

        if (distinctColors.size > 2) {
            score++;
            reasons.push(`Complex Colors (${distinctColors.size} clusters) (+1)`);
        }

        // 4. Internal irregularity (Standard Deviation of intensity)
        const meanInt = intVals.reduce((a, b) => a + b, 0) / (intVals.length || 1);
        const variance = intVals.reduce((a, b) => a + Math.pow(b - meanInt, 2), 0) / (intVals.length || 1);
        const stdDev = Math.sqrt(variance);

        if (stdDev > 25) { // High variance means not uniform
            score++;
            reasons.push(`Internal Irregularity (StdDev: ${stdDev.toFixed(1)}) (+1)`);
        }

        scoredLesions.push({
            id: index + 1,
            score,
            reasons: reasons.length > 0 ? reasons : ['No suspicious traits (-0)'],
            bounds: { x: globalX, y: globalY, w: lesion.w, h: lesion.h },
            pixels: lesion.pixels,
            offsetX: lesion.offsetX,
            offsetY: lesion.offsetY
        });
    });

    // Sort descending by score
    return scoredLesions.sort((a, b) => b.score - a.score);
}

function drawDetectionOverlay(detectedLesions) {
    // Redraw original and render circle-only detections.
    drawOriginalImage();

    originalCtx.lineWidth = 2;
    originalCtx.strokeStyle = 'rgba(59, 130, 246, 0.95)';

    detectedLesions.forEach((lesion, index) => {
        const globalX = lesion.offsetX + lesion.x;
        const globalY = lesion.offsetY + lesion.y;
        const centerX = globalX + lesion.w / 2;
        const centerY = globalY + lesion.h / 2;
        const radius = Math.max(6, Math.max(lesion.w, lesion.h) * 0.55);

        originalCtx.beginPath();
        originalCtx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        originalCtx.stroke();

        if (index < 10) {
            originalCtx.fillStyle = '#ffffff';
            originalCtx.font = 'bold 12px Inter';
            originalCtx.fillText(`L-${index + 1}`, globalX, globalY - 4);
        }
    });
}

function displayResults(lesions) {
    resultsList.innerHTML = '';

    if (lesions.length === 0) {
        resultsList.innerHTML = '<div class="empty-state">No structures detected in selection.</div>';
        return;
    }

    // Show top 5
    const top5 = lesions.slice(0, 5);

    top5.forEach(lesion => {
        const card = document.createElement('div');
        card.className = 'result-card';

        let scoreClass = 'score-low';
        if (lesion.score >= 3) scoreClass = 'score-high';
        else if (lesion.score == 2) scoreClass = 'score-med';

        const reasonsHtml = lesion.reasons.map(r => `<li>${r}</li>`).join('');

        card.innerHTML = `
            <div class="result-header">
                <span class="lesion-id">Lesion L-${lesion.id}</span>
                <span class="score-badge ${scoreClass}">Score: ${lesion.score}/4</span>
            </div>
            <div class="scoring-details">
                <ul>${reasonsHtml}</ul>
            </div>
        `;

        resultsList.appendChild(card);
    });
}
