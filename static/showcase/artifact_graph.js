/**
 * SAGE — Obsidian-Inspired Interactive 3D Artifact Graph
 * 
 * Scope & Architecture:
 * - Read/visualization layer over SAGE's canonical ArtifactStore (PDF, DOC/DOCX, Images, Sheets, Code)
 * - Uses existing doc_ids as node IDs and verified relationships as edges
 * - Zero external CDN requests — runs entirely offline using local three.min.js
 * - Optimized for 8 GB RAM laptops: force physics sleeps when settled, WebGL loop throttles when idle
 * - Reuses existing SAGE document preview modal (#docPreviewModal)
 * - Extensible: structured with NODE_SCHEMAS for future node types
 */

(function (global) {
    'use strict';

    // ── File Type Color Palette (Obsidian Cosmic Aesthetic) ───────────────────
    const TYPE_COLORS = {
        pdf:   { hex: 0xef4444, css: '#ef4444', label: 'PDF',   glow: 'rgba(239, 68, 68, 0.45)' },
        docx:  { hex: 0x3b82f6, css: '#3b82f6', label: 'DOCX',  glow: 'rgba(59, 130, 246, 0.45)' },
        doc:   { hex: 0x3b82f6, css: '#3b82f6', label: 'DOC',   glow: 'rgba(59, 130, 246, 0.45)' },
        png:   { hex: 0xa855f7, css: '#a855f7', label: 'IMAGE', glow: 'rgba(168, 85, 247, 0.45)' },
        jpg:   { hex: 0xa855f7, css: '#a855f7', label: 'IMAGE', glow: 'rgba(168, 85, 247, 0.45)' },
        jpeg:  { hex: 0xa855f7, css: '#a855f7', label: 'IMAGE', glow: 'rgba(168, 85, 247, 0.45)' },
        webp:  { hex: 0xa855f7, css: '#a855f7', label: 'IMAGE', glow: 'rgba(168, 85, 247, 0.45)' },
        gif:   { hex: 0xa855f7, css: '#a855f7', label: 'IMAGE', glow: 'rgba(168, 85, 247, 0.45)' },
        svg:   { hex: 0xa855f7, css: '#a855f7', label: 'IMAGE', glow: 'rgba(168, 85, 247, 0.45)' },
        csv:   { hex: 0x10b981, css: '#10b981', label: 'SHEET', glow: 'rgba(16, 185, 129, 0.45)' },
        xlsx:  { hex: 0x10b981, css: '#10b981', label: 'SHEET', glow: 'rgba(16, 185, 129, 0.45)' },
        xls:   { hex: 0x10b981, css: '#10b981', label: 'SHEET', glow: 'rgba(16, 185, 129, 0.45)' },
        py:    { hex: 0x06b6d4, css: '#06b6d4', label: 'CODE',  glow: 'rgba(6, 182, 212, 0.45)' },
        js:    { hex: 0x06b6d4, css: '#06b6d4', label: 'CODE',  glow: 'rgba(6, 182, 212, 0.45)' },
        ts:    { hex: 0x06b6d4, css: '#06b6d4', label: 'CODE',  glow: 'rgba(6, 182, 212, 0.45)' },
        html:  { hex: 0x06b6d4, css: '#06b6d4', label: 'CODE',  glow: 'rgba(6, 182, 212, 0.45)' },
        css:   { hex: 0x06b6d4, css: '#06b6d4', label: 'CODE',  glow: 'rgba(6, 182, 212, 0.45)' },
        json:  { hex: 0x06b6d4, css: '#06b6d4', label: 'CODE',  glow: 'rgba(6, 182, 212, 0.45)' },
        txt:   { hex: 0xf59e0b, css: '#f59e0b', label: 'TEXT',  glow: 'rgba(245, 158, 11, 0.45)' },
        md:    { hex: 0xf59e0b, css: '#f59e0b', label: 'TEXT',  glow: 'rgba(245, 158, 11, 0.45)' },
        other: { hex: 0x64748b, css: '#64748b', label: 'FILE',  glow: 'rgba(100, 116, 139, 0.45)' },
    };

    function getColorForType(ext) {
        const clean = (ext || '').toLowerCase().replace(/^\./, '');
        return TYPE_COLORS[clean] || TYPE_COLORS.other;
    }

    // Helper: generate circular glow texture for node bloom sprites
    function createGlowTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 64;
        const ctx = canvas.getContext('2d');
        const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
        grad.addColorStop(0.0, 'rgba(255, 255, 255, 1.0)');
        grad.addColorStop(0.2, 'rgba(255, 255, 255, 0.7)');
        grad.addColorStop(0.5, 'rgba(255, 255, 255, 0.25)');
        grad.addColorStop(1.0, 'rgba(255, 255, 255, 0.0)');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 64, 64);
        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;
        return texture;
    }

    // Helper: generate crisp text sprite for node billboard labels
    function createTextSprite(text, colorCss) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const fontSize = 24;
        ctx.font = `600 ${fontSize}px Inter, system-ui, sans-serif`;

        // Truncate long filename for 3D label
        let display = text;
        if (display.length > 20) {
            display = display.substring(0, 10) + '…' + display.substring(display.length - 8);
        }

        const metrics = ctx.measureText(display);
        const textWidth = Math.ceil(metrics.width);
        const padding = 16;
        canvas.width = textWidth + padding * 2;
        canvas.height = fontSize + padding * 2;

        // Redraw after resizing canvas
        ctx.font = `600 ${fontSize}px Inter, system-ui, sans-serif`;
        ctx.fillStyle = 'rgba(8, 14, 11, 0.82)';
        ctx.beginPath();
        const r = 8;
        const w = canvas.width, h = canvas.height;
        ctx.roundRect ? ctx.roundRect(0, 0, w, h, r) : ctx.rect(0, 0, w, h);
        ctx.fill();

        ctx.strokeStyle = colorCss || 'rgba(37, 99, 235, 0.4)';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = '#f8fafc';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(display, canvas.width / 2, canvas.height / 2);

        const texture = new THREE.CanvasTexture(canvas);
        texture.minFilter = THREE.LinearFilter;
        const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
        const sprite = new THREE.Sprite(material);
        const aspect = canvas.width / canvas.height;
        const baseH = 5.0;
        sprite.scale.set(baseH * aspect, baseH, 1.0);
        return sprite;
    }

    // ── 3D Artifact Graph Class ───────────────────────────────────────────────
    class SageArtifactGraph {
        constructor(container, options = {}) {
            this.container = container;
            this.options = Object.assign({
                onNodeClick: null,
                onNodeDblClick: null,
                onNodeHover: null,
            }, options);

            let canvasEl = container.querySelector('#artifactGraphCanvas');
            if (canvasEl && canvasEl.tagName.toLowerCase() !== 'canvas') {
                let existingCanvas = canvasEl.querySelector('canvas');
                if (!existingCanvas) {
                    existingCanvas = document.createElement('canvas');
                    existingCanvas.style.width = '100%';
                    existingCanvas.style.height = '100%';
                    existingCanvas.style.display = 'block';
                    canvasEl.appendChild(existingCanvas);
                }
                this.canvasContainer = canvasEl;
                this.canvas = existingCanvas;
            } else {
                this.canvas = canvasEl || document.createElement('canvas');
                this.canvasContainer = container;
            }

            this.tooltip = container.querySelector('#artifactGraphTooltip') || container.querySelector('#graphTooltip');
            this.emptyState = container.querySelector('#graphEmptyState');
            this.counterEl = container.querySelector('#graphNodeCount');
            this.edgeCounterEl = container.querySelector('#graphEdgeCount');

            // Data State
            this.nodesData = [];
            this.edgesData = [];
            this.summary = {};
            this.nodesMap = new Map();   // id -> nodeData
            this.nodeMeshes = [];        // array of THREE.Mesh
            this.edgeLines = null;       // THREE.LineSegments

            // Filter State
            this.activeTypeFilter = 'all';
            this.activeSearch = '';
            this.onlyConnected = false;
            this.selectedNodeId = null;
            this.hoveredNode = null;

            // Physics state
            this.isSimulating = true;
            this.isAwake = true;
            this.simulationTicks = 0;
            this.maxSimulationTicks = 450; // auto-sleep after settling

            // Three.js instances
            this.scene = null;
            this.camera = null;
            this.renderer = null;
            this.raycaster = new THREE.Raycaster();
            this.pointer = new THREE.Vector2(-999, -999);
            this.glowTexture = null;

            // Camera control state (Custom smooth orbit/pan/zoom)
            this.camTarget = new THREE.Vector3(0, 0, 0);
            this.camPos = new THREE.Vector3(0, 45, 140);
            this.camDistance = 140;
            this.camTheta = 0;
            this.camPhi = Math.PI / 3.8;
            this.targetCamDistance = 140;
            this.targetCamTheta = 0;
            this.targetCamPhi = Math.PI / 3.8;
            this.targetLookAt = new THREE.Vector3(0, 0, 0);

            // Interaction dragging state
            this.isDragging = false;
            this.dragMode = 'orbit'; // 'orbit' or 'pan'
            this.dragStart = { x: 0, y: 0 };
            this.lastPointerPos = { x: 0, y: 0 };
            this.animFrameId = null;
            this.isActive = false;

            this._initThree();
            this._bindEvents();
            this._initHUD();
        }

        _initThree() {
            if (!this.canvas) return;

            // Scene
            this.scene = new THREE.Scene();
            this.scene.fog = new THREE.FogExp2(0x070b09, 0.0022);

            // Camera
            const rect = this.container.getBoundingClientRect();
            const width = rect.width || 800;
            const height = rect.height || 600;
            this.camera = new THREE.PerspectiveCamera(55, width / height, 1, 3000);
            this._updateCameraFromOrbit();

            // Renderer
            this.renderer = new THREE.WebGLRenderer({
                canvas: this.canvas,
                antialias: true,
                alpha: true,
                powerPreference: 'high-performance',
            });
            this.renderer.setSize(width, height);
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
            this.renderer.setClearColor(0x070b09, 1.0);

            // Lights
            const ambient = new THREE.AmbientLight(0xffffff, 0.85);
            this.scene.add(ambient);

            const dirLight = new THREE.DirectionalLight(0x2ca0bd, 0.7);
            dirLight.position.set(60, 100, 80);
            this.scene.add(dirLight);

            const dirLight2 = new THREE.DirectionalLight(0x186c82, 0.45);
            dirLight2.position.set(-60, -80, -50);
            this.scene.add(dirLight2);

            // Background Cosmic Star Particles
            this._createStarfield();

            // Shared glow texture for performance
            this.glowTexture = createGlowTexture();
        }

        _createStarfield() {
            const count = 450;
            const geom = new THREE.BufferGeometry();
            const positions = new Float32Array(count * 3);
            const colors = new Float32Array(count * 3);

            for (let i = 0; i < count; i++) {
                const r = 250 + Math.random() * 550;
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.acos((Math.random() * 2) - 1);
                positions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
                positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
                positions[i * 3 + 2] = r * Math.cos(phi);

                // Subtle blended teal-blue and pearl star dust
                const tint = Math.random();
                if (tint < 0.45) {
                    // Soft glacier teal-blue (#2085a0)
                    colors[i * 3] = 0.145; colors[i * 3 + 1] = 0.631; colors[i * 3 + 2] = 0.761;
                } else if (tint < 0.8) {
                    // Light mist (#2ca0bd)
                    colors[i * 3] = 0.17; colors[i * 3 + 1] = 0.63; colors[i * 3 + 2] = 0.74;
                } else {
                    // Soft pearl
                    colors[i * 3] = 0.85; colors[i * 3 + 1] = 0.90; colors[i * 3 + 2] = 0.92;
                }
            }

            geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));

            const mat = new THREE.PointsMaterial({
                size: 2.2,
                vertexColors: true,
                transparent: true,
                opacity: 0.55,
                depthWrite: false,
            });
            const stars = new THREE.Points(geom, mat);
            this.scene.add(stars);
            this.starsMesh = stars;
        }

        _bindEvents() {
            window.addEventListener('resize', () => this.handleResize());

            // Pointer down (Orbit or Pan)
            this.canvas.addEventListener('mousedown', (e) => {
                this.isDragging = true;
                this.dragStart = { x: e.clientX, y: e.clientY };
                this.lastPointerPos = { x: e.clientX, y: e.clientY };
                this.dragMode = (e.button === 2 || e.shiftKey || e.button === 1) ? 'pan' : 'orbit';
                this.wakeSimulation();
            });

            // Context menu disable on canvas for right click pan
            this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

            // Pointer move (Orbit, Pan, Raycasting)
            window.addEventListener('mousemove', (e) => {
                const rect = this.canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;

                // Update normalized pointer for raycasting if within canvas
                if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {
                    this.pointer.x = (x / rect.width) * 2 - 1;
                    this.pointer.y = -(y / rect.height) * 2 + 1;
                } else {
                    this.pointer.x = -999;
                    this.pointer.y = -999;
                }

                if (this.isDragging) {
                    const dx = e.clientX - this.lastPointerPos.x;
                    const dy = e.clientY - this.lastPointerPos.y;
                    this.lastPointerPos = { x: e.clientX, y: e.clientY };

                    if (this.dragMode === 'orbit') {
                        this.targetCamTheta -= dx * 0.007;
                        this.targetCamPhi = Math.max(0.08, Math.min(Math.PI - 0.08, this.targetCamPhi - dy * 0.007));
                    } else if (this.dragMode === 'pan') {
                        // Pan orthogonal to camera look
                        const factor = this.camDistance * 0.0014;
                        const forward = new THREE.Vector3().subVectors(this.camTarget, this.camera.position).normalize();
                        const right = new THREE.Vector3().crossVectors(forward, this.camera.up).normalize();
                        const up = this.camera.up.clone().normalize();

                        this.targetLookAt.addScaledVector(right, -dx * factor);
                        this.targetLookAt.addScaledVector(up, dy * factor);
                    }
                    this.wakeSimulation();
                } else {
                    this._handlePointerHover(e);
                }
            });

            // Pointer up
            window.addEventListener('mouseup', (e) => {
                if (this.isDragging) {
                    const dist = Math.hypot(e.clientX - this.dragStart.x, e.clientY - this.dragStart.y);
                    this.isDragging = false;
                    // If pointer barely moved, treat as click
                    if (dist < 4) {
                        this._handlePointerClick(e);
                    }
                }
            });

            // Wheel zoom
            this.canvas.addEventListener('wheel', (e) => {
                e.preventDefault();
                const zoomFactor = e.deltaY > 0 ? 1.12 : 0.89;
                this.targetCamDistance = Math.max(25, Math.min(650, this.targetCamDistance * zoomFactor));
                this.wakeSimulation();
            }, { passive: false });

            // Double click to focus and preview
            this.canvas.addEventListener('dblclick', (e) => {
                if (this.hoveredNode) {
                    this.focusNode(this.hoveredNode.userData.id, true);
                    if (this.options.onNodeDblClick) {
                        this.options.onNodeDblClick(this.hoveredNode.userData);
                    }
                }
            });

            // Touch events for mobile/tablet orbit & pinch zoom
            let initialPinchDist = 0;
            this.canvas.addEventListener('touchstart', (e) => {
                if (e.touches.length === 1) {
                    this.isDragging = true;
                    this.dragStart = { x: e.touches[0].clientX, y: e.touches[0].clientY };
                    this.lastPointerPos = { x: e.touches[0].clientX, y: e.touches[0].clientY };
                    this.dragMode = 'orbit';
                } else if (e.touches.length === 2) {
                    this.isDragging = false;
                    initialPinchDist = Math.hypot(
                        e.touches[0].clientX - e.touches[1].clientX,
                        e.touches[0].clientY - e.touches[1].clientY
                    );
                }
                this.wakeSimulation();
            }, { passive: true });

            this.canvas.addEventListener('touchmove', (e) => {
                if (e.touches.length === 1 && this.isDragging) {
                    const dx = e.touches[0].clientX - this.lastPointerPos.x;
                    const dy = e.touches[0].clientY - this.lastPointerPos.y;
                    this.lastPointerPos = { x: e.touches[0].clientX, y: e.touches[0].clientY };
                    this.targetCamTheta -= dx * 0.007;
                    this.targetCamPhi = Math.max(0.08, Math.min(Math.PI - 0.08, this.targetCamPhi - dy * 0.007));
                    this.wakeSimulation();
                } else if (e.touches.length === 2) {
                    const currentDist = Math.hypot(
                        e.touches[0].clientX - e.touches[1].clientX,
                        e.touches[0].clientY - e.touches[1].clientY
                    );
                    if (initialPinchDist > 0) {
                        const ratio = initialPinchDist / currentDist;
                        this.targetCamDistance = Math.max(25, Math.min(650, this.targetCamDistance * (ratio > 1 ? 1.03 : 0.97)));
                        initialPinchDist = currentDist;
                    }
                    this.wakeSimulation();
                }
            }, { passive: true });

            this.canvas.addEventListener('touchend', () => {
                this.isDragging = false;
                initialPinchDist = 0;
            }, { passive: true });
        }

        _initHUD() {
            // Search filter
            const searchInput = this.container.querySelector('#graphSearchInput');
            if (searchInput) {
                searchInput.addEventListener('input', (e) => {
                    this.activeSearch = e.target.value.trim();
                    this._applyVisualFilters();
                    this.wakeSimulation();
                });
            }

            // File type filter pills
            const typePills = this.container.querySelectorAll('.graph-pill');
            typePills.forEach(pill => {
                pill.addEventListener('click', () => {
                    typePills.forEach(p => p.classList.remove('active'));
                    pill.classList.add('active');
                    this.activeTypeFilter = pill.dataset.type || 'all';
                    this._applyVisualFilters();
                    this.wakeSimulation();
                });
            });

            // Connected only toggle
            const connToggle = this.container.querySelector('#graphConnectedToggle');
            if (connToggle) {
                connToggle.addEventListener('change', (e) => {
                    this.onlyConnected = e.target.checked;
                    this._applyVisualFilters();
                    this.wakeSimulation();
                });
            }

            // Reset camera button
            const resetBtn = this.container.querySelector('#graphResetViewBtn') || this.container.querySelector('#graphResetBtn');
            if (resetBtn) {
                resetBtn.addEventListener('click', () => this.resetCamera());
            }

            // Focus selected button
            const focusBtn = this.container.querySelector('#graphFocusSelectedBtn') || this.container.querySelector('#graphFocusBtn');
            if (focusBtn) {
                focusBtn.addEventListener('click', () => {
                    if (this.selectedNodeId) {
                        this.focusNode(this.selectedNodeId);
                    } else if (this.nodeMeshes.length > 0) {
                        this.focusNode(this.nodeMeshes[0].userData.id);
                    }
                });
            }

            // Refresh button
            const refreshBtn = this.container.querySelector('#graphRefreshBtn');
            if (refreshBtn) {
                refreshBtn.addEventListener('click', () => this.loadData());
            }

            // Add Files button
            const addFilesBtn = this.container.querySelector('#graphAddFilesBtn');
            const fileInput = document.getElementById('artifactFileInput') || document.getElementById('fileInput');
            if (addFilesBtn && fileInput) {
                addFilesBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    fileInput.value = '';
                    fileInput.click();
                });
            }

            // Empty state upload button
            const uploadBtn = this.container.querySelector('#graphUploadPromptBtn') || this.container.querySelector('#graphUploadBtn');
            if (uploadBtn && fileInput) {
                uploadBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    fileInput.value = '';
                    fileInput.click();
                });
            }

            // Delete Selected button
            const deleteBtn = this.container.querySelector('#graphDeleteBtn');
            if (deleteBtn) {
                deleteBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    if (this.selectedNodeId) {
                        this.deleteNode(this.selectedNodeId);
                    }
                });
            }

            // Keyboard shortcut for delete
            window.addEventListener('keydown', (e) => {
                if ((e.key === 'Delete' || e.key === 'Backspace') && this.selectedNodeId) {
                    const tag = document.activeElement?.tagName?.toLowerCase();
                    if (tag === 'input' || tag === 'textarea') return;
                    e.preventDefault();
                    this.deleteNode(this.selectedNodeId);
                }
            });
        }

        // ── Camera Orbit Math ─────────────────────────────────────────────────
        _updateCameraFromOrbit() {
            if (!this.camera) return;
            const x = this.camTarget.x + this.camDistance * Math.sin(this.camPhi) * Math.sin(this.camTheta);
            const y = this.camTarget.y + this.camDistance * Math.cos(this.camPhi);
            const z = this.camTarget.z + this.camDistance * Math.sin(this.camPhi) * Math.cos(this.camTheta);
            this.camera.position.set(x, y, z);
            this.camera.lookAt(this.camTarget);
        }

        resetCamera() {
            this.targetCamDistance = 140;
            this.targetCamTheta = 0;
            this.targetCamPhi = Math.PI / 3.8;
            this.targetLookAt.set(0, 0, 0);
            this.selectedNodeId = null;
            const delBtn = this.container.querySelector('#graphDeleteBtn');
            if (delBtn) delBtn.style.display = 'none';
            this._applyVisualFilters();
            this.wakeSimulation();
        }

        focusNode(nodeId, immediate = false) {
            const mesh = this.nodeMeshes.find(m => m.userData.id === nodeId);
            if (!mesh) return;

            this.selectedNodeId = nodeId;
            const delBtn = this.container.querySelector('#graphDeleteBtn');
            if (delBtn) delBtn.style.display = 'inline-flex';

            this.targetLookAt.copy(mesh.position);
            this.targetCamDistance = 48; // Zoom in comfortably close

            if (immediate) {
                this.camTarget.copy(this.targetLookAt);
                this.camDistance = this.targetCamDistance;
                this._updateCameraFromOrbit();
            }
            this._applyVisualFilters();
            this.wakeSimulation();
        }

        async deleteNode(nodeId) {
            if (!nodeId) return;
            const node = this.nodesMap.get(nodeId) || (this.nodesData || []).find(n => n.id === nodeId);
            const label = node ? (node.label || node.name || 'document') : 'document';

            if (!confirm(`Are you sure you want to delete "${label}"?`)) {
                return;
            }

            // Remove from graph node and edge arrays
            const updatedNodes = (this.nodesData || []).filter(n => n.id !== nodeId);
            const updatedEdges = (this.edgesData || []).filter(e => {
                const s = typeof e.source === 'object' ? e.source.id : e.source;
                const t = typeof e.target === 'object' ? e.target.id : e.target;
                return s !== nodeId && t !== nodeId;
            });

            if (this.selectedNodeId === nodeId) {
                this.selectedNodeId = null;
                const delBtn = this.container.querySelector('#graphDeleteBtn');
                if (delBtn) delBtn.style.display = 'none';
            }

            if (this.tooltip) {
                this.tooltip.style.display = 'none';
            }

            this.setData(updatedNodes, updatedEdges, this.summary);
            this.wakeSimulation();

            if (this.options.onNodeDelete) {
                this.options.onNodeDelete(nodeId);
            }

            // Delete from backend if active
            try {
                await fetch(`/api/artifacts/${encodeURIComponent(nodeId)}`, { method: 'DELETE' });
            } catch (err) {
                console.log('Backend offline or local node deleted:', err.message);
            }
        }

        // ── Data Loading & Graph Construction ─────────────────────────────────

        async loadData() {
            try {
                const res = await fetch('/api/artifacts/graph');
                if (!res.ok) throw new Error('Failed to fetch artifact graph data');
                const data = await res.json();
                // Merge with any locally added files
                const localNodes = (this.nodesData || []).filter(n => n.id && (String(n.id).startsWith('user_art_') || String(n.id).startsWith('art_local_')));
                const backendNodes = data.nodes || [];
                const mergedNodes = [...localNodes, ...backendNodes.filter(bn => !localNodes.some(ln => ln.id === bn.id))];
                this.setData(mergedNodes.length ? mergedNodes : backendNodes, data.edges || [], data.summary || {});
            } catch (err) {
                console.warn('Backend graph API unavailable (offline / standalone mode):', err.message);
                // Do NOT wipe nodes if the user has already added local files
                if (!this.nodesData || this.nodesData.length === 0) {
                    this.setData([], [], {});
                }
            }
        }

        setData(nodes, edges, summary = {}) {
            this.nodesData = nodes || [];
            this.edgesData = edges || [];
            this.summary = summary || {};
            this.nodesMap.clear();

            // Check empty state
            const hasNodes = this.nodesData.length > 0;
            if (this.emptyState) {
                this.emptyState.style.display = hasNodes ? 'none' : 'flex';
            }
            if (this.counterEl) {
                this.counterEl.textContent = `${this.nodesData.length} artifact${this.nodesData.length === 1 ? '' : 's'}`;
            }
            if (this.edgeCounterEl) {
                this.edgeCounterEl.textContent = `${this.edgesData.length} relation${this.edgesData.length === 1 ? '' : 's'}`;
            }

            this._clearGraph();
            if (!hasNodes) return;

            // 1. Initial 3D Spherical Coordinate Placement
            const n = nodes.length;
            const goldenRatio = (1 + Math.sqrt(5)) / 2;
            const spreadRadius = Math.max(30, Math.sqrt(n) * 22);

            nodes.forEach((node, i) => {
                // Ensure required schema attributes exist
                node.label = node.label || node.name || node.title || 'Untitled';
                node.file_type = (node.file_type || node.type || (node.label.split('.').pop() || '')).toLowerCase();
                node.degree = typeof node.degree === 'number' ? node.degree : (node.connected_count || 0);

                const theta = 2 * Math.PI * i / goldenRatio;
                const phi = Math.acos(1 - 2 * (i + 0.5) / n);
                const r = spreadRadius * (0.6 + Math.random() * 0.4);

                node.x = r * Math.sin(phi) * Math.cos(theta);
                node.y = r * Math.sin(phi) * Math.sin(theta);
                node.z = r * Math.cos(phi);
                node.vx = 0;
                node.vy = 0;
                node.vz = 0;

                this.nodesMap.set(node.id, node);
            });

            // 2. Build 3D Meshes for Nodes
            nodes.forEach(node => {
                const palette = getColorForType(node.file_type);
                const baseRadius = 3.6 + Math.min(node.degree * 1.5, 10);

                // Core Sphere
                const geom = new THREE.SphereGeometry(baseRadius, 24, 24);
                const mat = new THREE.MeshStandardMaterial({
                    color: palette.hex,
                    emissive: palette.hex,
                    emissiveIntensity: 0.45,
                    roughness: 0.25,
                    metalness: 0.2,
                });
                const mesh = new THREE.Mesh(geom, mat);
                mesh.position.set(node.x, node.y, node.z);
                mesh.userData = node;

                // Soft Outer Halo Bloom Sprite
                const haloMat = new THREE.SpriteMaterial({
                    map: this.glowTexture,
                    color: palette.hex,
                    transparent: true,
                    opacity: 0.65,
                    blending: THREE.AdditiveBlending,
                    depthWrite: false,
                });
                const haloSprite = new THREE.Sprite(haloMat);
                const haloSize = baseRadius * 3.8;
                haloSprite.scale.set(haloSize, haloSize, 1.0);
                mesh.add(haloSprite);

                // Billboard Label
                const labelSprite = createTextSprite(node.label, palette.css);
                labelSprite.position.set(0, baseRadius + 3.8, 0);
                mesh.add(labelSprite);

                mesh.halo = haloSprite;
                mesh.label = labelSprite;
                mesh.baseRadius = baseRadius;

                this.scene.add(mesh);
                this.nodeMeshes.push(mesh);
            });

            // 3. Build LineSegments for Edges
            this._buildEdgeGeometry();

            // Start physics simulation
            this.simulationTicks = 0;
            this.wakeSimulation();
            this._applyVisualFilters();
        }

        _clearGraph() {
            this.nodeMeshes.forEach(mesh => {
                if (mesh.geometry) mesh.geometry.dispose();
                if (mesh.material) mesh.material.dispose();
                if (mesh.halo) mesh.halo.material.dispose();
                if (mesh.label) {
                    if (mesh.label.material.map) mesh.label.material.map.dispose();
                    mesh.label.material.dispose();
                }
                this.scene.remove(mesh);
            });
            this.nodeMeshes = [];

            if (this.edgeLines) {
                if (this.edgeLines.geometry) this.edgeLines.geometry.dispose();
                if (this.edgeLines.material) this.edgeLines.material.dispose();
                this.scene.remove(this.edgeLines);
                this.edgeLines = null;
            }
        }

        _buildEdgeGeometry() {
            if (this.edgeLines) {
                this.scene.remove(this.edgeLines);
                this.edgeLines = null;
            }

            const validEdges = this.edgesData.filter(e =>
                this.nodesMap.has(e.source) && this.nodesMap.has(e.target)
            );

            if (!validEdges.length) return;

            const positions = new Float32Array(validEdges.length * 2 * 3);
            const colors = new Float32Array(validEdges.length * 2 * 3);

            validEdges.forEach((edge, i) => {
                const src = this.nodesMap.get(edge.source);
                const tgt = this.nodesMap.get(edge.target);

                positions[i * 6]     = src.x;
                positions[i * 6 + 1] = src.y;
                positions[i * 6 + 2] = src.z;
                positions[i * 6 + 3] = tgt.x;
                positions[i * 6 + 4] = tgt.y;
                positions[i * 6 + 5] = tgt.z;

                const c1 = getColorForType(src.file_type);
                const c2 = getColorForType(tgt.file_type);

                const col1 = new THREE.Color(c1.hex);
                const col2 = new THREE.Color(c2.hex);

                colors[i * 6]     = col1.r; colors[i * 6 + 1] = col1.g; colors[i * 6 + 2] = col1.b;
                colors[i * 6 + 3] = col2.r; colors[i * 6 + 4] = col2.g; colors[i * 6 + 5] = col2.b;
            });

            const geom = new THREE.BufferGeometry();
            geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));

            const mat = new THREE.LineBasicMaterial({
                vertexColors: true,
                transparent: true,
                opacity: 0.38,
                blending: THREE.AdditiveBlending,
                depthWrite: false,
            });

            this.edgeLines = new THREE.LineSegments(geom, mat);
            this.edgeLines.userData = { validEdges };
            this.scene.add(this.edgeLines);
        }

        // ── 3D Force-Directed Simulation (Coulomb + Hooke + Center Gravity) ───

        _stepPhysics() {
            if (this.simulationTicks >= this.maxSimulationTicks) {
                this.isSimulating = false;
                return;
            }

            const nodes = this.nodesData;
            const n = nodes.length;
            if (n === 0) return;

            // 1. Coulomb Repulsion (all node pairs)
            const kRepulse = 380;
            for (let i = 0; i < n; i++) {
                const a = nodes[i];
                for (let j = i + 1; j < n; j++) {
                    const b = nodes[j];
                    const dx = a.x - b.x;
                    const dy = a.y - b.y;
                    const dz = a.z - b.z;
                    const distSq = dx * dx + dy * dy + dz * dz + 40;
                    const dist = Math.sqrt(distSq);
                    const force = kRepulse / distSq;

                    const fx = (dx / dist) * force;
                    const fy = (dy / dist) * force;
                    const fz = (dz / dist) * force;

                    a.vx += fx; a.vy += fy; a.vz += fz;
                    b.vx -= fx; b.vy -= fy; b.vz -= fz;
                }
            }

            // 2. Hooke Spring Attraction (connected edges)
            const kSpring = 0.035;
            const targetLength = 36;
            this.edgesData.forEach(edge => {
                const src = this.nodesMap.get(edge.source);
                const tgt = this.nodesMap.get(edge.target);
                if (!src || !tgt) return;

                const dx = tgt.x - src.x;
                const dy = tgt.y - src.y;
                const dz = tgt.z - src.z;
                const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.001;
                const displacement = dist - targetLength;
                const force = kSpring * displacement * (edge.weight || 1.0);

                const fx = (dx / dist) * force;
                const fy = (dy / dist) * force;
                const fz = (dz / dist) * force;

                src.vx += fx; src.vy += fy; src.vz += fz;
                tgt.vx -= fx; tgt.vy -= fy; tgt.vz -= fz;
            });

            // 3. Center Gravity & Damping
            const kCenter = 0.008;
            const damping = 0.86;
            let totalKineticEnergy = 0;

            nodes.forEach(node => {
                node.vx -= node.x * kCenter;
                node.vy -= node.y * kCenter;
                node.vz -= node.z * kCenter;

                node.vx *= damping;
                node.vy *= damping;
                node.vz *= damping;

                node.x += node.vx;
                node.y += node.vy;
                node.z += node.vz;

                totalKineticEnergy += (node.vx * node.vx + node.vy * node.vy + node.vz * node.vz);
            });

            // Update 3D Mesh positions
            this.nodeMeshes.forEach(mesh => {
                const d = mesh.userData;
                mesh.position.set(d.x, d.y, d.z);
            });

            // Update edge line endpoints
            if (this.edgeLines && this.edgeLines.geometry) {
                const posAttr = this.edgeLines.geometry.attributes.position;
                const edges = this.edgeLines.userData.validEdges || [];
                edges.forEach((edge, i) => {
                    const src = this.nodesMap.get(edge.source);
                    const tgt = this.nodesMap.get(edge.target);
                    if (src && tgt) {
                        posAttr.setXYZ(i * 2,     src.x, src.y, src.z);
                        posAttr.setXYZ(i * 2 + 1, tgt.x, tgt.y, tgt.z);
                    }
                });
                posAttr.needsUpdate = true;
            }

            this.simulationTicks++;
            if (totalKineticEnergy < 0.008) {
                this.isSimulating = false;
            }
        }

        wakeSimulation() {
            this.isSimulating = true;
            this.isAwake = true;
            this.simulationTicks = 0;
        }

        // ── Visual Filtering & Search ─────────────────────────────────────────

        _applyVisualFilters() {
            const hasSearch = this.activeSearch.length > 0;
            const searchLower = this.activeSearch.toLowerCase();
            const filterType = this.activeTypeFilter;

            let visibleCount = 0;

            this.nodeMeshes.forEach(mesh => {
                const d = mesh.userData;
                let matchesType = filterType === 'all' || d.file_type === filterType;
                let matchesSearch = !hasSearch ||
                    d.label.toLowerCase().includes(searchLower) ||
                    d.id.toLowerCase().includes(searchLower);
                let matchesConnected = !this.onlyConnected || d.degree > 0;

                const isVisible = matchesType && matchesSearch && matchesConnected;

                if (isVisible) {
                    visibleCount++;
                    mesh.visible = true;
                    // If a node is selected, highlight only it and its neighbors
                    if (this.selectedNodeId) {
                        const isSelected = d.id === this.selectedNodeId;
                        const isNeighbor = this._areNodesConnected(d.id, this.selectedNodeId);
                        if (isSelected || isNeighbor) {
                            mesh.material.opacity = 1.0;
                            mesh.material.emissiveIntensity = isSelected ? 0.9 : 0.55;
                            if (mesh.halo) mesh.halo.material.opacity = isSelected ? 0.95 : 0.55;
                            if (mesh.label) mesh.label.material.opacity = 1.0;
                        } else {
                            mesh.material.opacity = 0.2;
                            mesh.material.emissiveIntensity = 0.1;
                            if (mesh.halo) mesh.halo.material.opacity = 0.1;
                            if (mesh.label) mesh.label.material.opacity = 0.15;
                        }
                    } else {
                        mesh.material.opacity = 1.0;
                        mesh.material.emissiveIntensity = 0.45;
                        if (mesh.halo) mesh.halo.material.opacity = 0.65;
                        if (mesh.label) mesh.label.material.opacity = 0.9;
                    }
                } else {
                    mesh.visible = false;
                }
            });

            // Update counter with filtered view count
            if (this.counterEl) {
                this.counterEl.textContent = `${visibleCount} of ${this.nodesData.length} artifact${this.nodesData.length === 1 ? '' : 's'} visible`;
            }
        }

        _areNodesConnected(id1, id2) {
            return this.edgesData.some(e =>
                (e.source === id1 && e.target === id2) || (e.source === id2 && e.target === id1)
            );
        }

        // ── Raycasting & Interaction ──────────────────────────────────────────

        _handlePointerHover(e) {
            if (!this.camera || !this.nodeMeshes.length) return;

            this.raycaster.setFromCamera(this.pointer, this.camera);
            const visibleMeshes = this.nodeMeshes.filter(m => m.visible);
            const intersects = this.raycaster.intersectObjects(visibleMeshes, false);

            if (intersects.length > 0) {
                const hit = intersects[0].object;
                if (this.hoveredNode !== hit) {
                    this._unhighlightHovered();
                    this.hoveredNode = hit;
                    this._highlightHovered(hit, e);
                } else {
                    this._positionTooltip(e);
                }
                this.canvas.style.cursor = 'pointer';
            } else {
                if (this.hoveredNode) {
                    this._unhighlightHovered();
                    this.hoveredNode = null;
                }
                this.canvas.style.cursor = 'default';
            }
        }

        _highlightHovered(mesh, event) {
            mesh.scale.set(1.22, 1.22, 1.22);
            if (mesh.halo) mesh.halo.material.opacity = 0.95;

            // Highlight incident edges
            if (this.edgeLines) {
                this.edgeLines.material.opacity = 0.85;
            }

            // Show and position tooltip
            this._renderTooltip(mesh.userData);
            this._positionTooltip(event);
            if (this.tooltip) this.tooltip.style.display = 'block';

            if (this.options.onNodeHover) {
                this.options.onNodeHover(mesh.userData);
            }
        }

        _unhighlightHovered() {
            if (this.hoveredNode) {
                this.hoveredNode.scale.set(1.0, 1.0, 1.0);
                if (this.hoveredNode.halo) {
                    this.hoveredNode.halo.material.opacity = 0.65;
                }
            }
            if (this.edgeLines) {
                this.edgeLines.material.opacity = 0.38;
            }
            if (this.tooltip) this.tooltip.style.display = 'none';
        }

        _positionTooltip(event) {
            if (!this.tooltip) return;
            const containerRect = this.container.getBoundingClientRect();
            let x = event.clientX - containerRect.left + 16;
            let y = event.clientY - containerRect.top + 16;

            const tipW = this.tooltip.offsetWidth || 240;
            const tipH = this.tooltip.offsetHeight || 130;

            if (x + tipW > containerRect.width - 20) {
                x = event.clientX - containerRect.left - tipW - 14;
            }
            if (y + tipH > containerRect.height - 20) {
                y = event.clientY - containerRect.top - tipH - 14;
            }

            this.tooltip.style.left = `${Math.max(10, x)}px`;
            this.tooltip.style.top = `${Math.max(10, y)}px`;
        }

        _renderTooltip(node) {
            if (!this.tooltip) return;
            const palette = getColorForType(node.file_type);
            const sizeKB = (node.size_bytes / 1024).toFixed(1);
            const sizeStr = node.size_bytes > 1024 * 1024 ? (node.size_bytes / (1024 * 1024)).toFixed(1) + ' MB' : sizeKB + ' KB';

            let pageText = '';
            if (node.metadata?.page_count) {
                pageText = `${node.metadata.page_count} page${node.metadata.page_count === 1 ? '' : 's'}`;
            } else if (node.metadata?.total_elements) {
                pageText = `${node.metadata.total_elements} element${node.metadata.total_elements === 1 ? '' : 's'}`;
            }

            this.tooltip.innerHTML = `
                <div class="gt-head">
                    <span class="gt-badge" style="background:${palette.glow};color:${palette.css};border-color:${palette.css}">
                        ${palette.label}
                    </span>
                    <span class="gt-degree">${node.degree} relationship${node.degree === 1 ? '' : 's'}</span>
                </div>
                <div class="gt-name" title="${node.label}">${node.label}</div>
                <div class="gt-meta">
                    <span>${sizeStr}</span>
                    ${pageText ? `<span>· ${pageText}</span>` : ''}
                    <span class="gt-id">ID: ${node.id.substring(0, 12)}…</span>
                </div>
                <div class="gt-actions">
                    <button class="gt-action-btn gt-preview-btn" title="Open document preview">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                        <span>Preview</span>
                    </button>
                    <button class="gt-action-btn gt-del-btn" title="Delete document">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                        <span>Delete</span>
                    </button>
                </div>
            `;

            const pBtn = this.tooltip.querySelector('.gt-preview-btn');
            if (pBtn) {
                pBtn.onclick = (e) => {
                    e.stopPropagation();
                    if (this.options.onNodeDblClick) {
                        this.options.onNodeDblClick(node);
                    }
                };
            }

            const dBtn = this.tooltip.querySelector('.gt-del-btn');
            if (dBtn) {
                dBtn.onclick = (e) => {
                    e.stopPropagation();
                    this.deleteNode(node.id);
                };
            }
        }

        _handlePointerClick(e) {
            if (this.hoveredNode) {
                const nodeData = this.hoveredNode.userData;
                this.focusNode(nodeData.id);
                if (this.options.onNodeClick) {
                    this.options.onNodeClick(nodeData);
                }
            } else {
                // Click on empty space clears selection
                if (this.selectedNodeId) {
                    this.selectedNodeId = null;
                    const delBtn = this.container.querySelector('#graphDeleteBtn');
                    if (delBtn) delBtn.style.display = 'none';
                    this._applyVisualFilters();
                }
            }
        }

        // ── Lifecycle & Render Loop ───────────────────────────────────────────

        start() {
            if (this.isActive) return;
            this.isActive = true;
            this.wakeSimulation();
            this.handleResize();
            this._tick();
        }

        stop() {
            this.isActive = false;
            if (this.animFrameId) {
                cancelAnimationFrame(this.animFrameId);
                this.animFrameId = null;
            }
        }

        _tick() {
            if (!this.isActive) return;

            // Camera smooth interpolation (slerp/lerp towards targets)
            const ease = 0.08;
            this.camDistance += (this.targetCamDistance - this.camDistance) * ease;
            this.camTheta += (this.targetCamTheta - this.camTheta) * ease;
            this.camPhi += (this.targetCamPhi - this.camPhi) * ease;
            this.camTarget.lerp(this.targetLookAt, ease);

            this._updateCameraFromOrbit();

            // Run physics simulation step if active
            if (this.isSimulating) {
                this._stepPhysics();
            }

            // Slowly rotate cosmic starfield for alive background feel
            if (this.starsMesh) {
                this.starsMesh.rotation.y += 0.0003;
                this.starsMesh.rotation.x += 0.0001;
            }

            // Render 3D Frame
            if (this.renderer && this.scene && this.camera) {
                this.renderer.render(this.scene, this.camera);
            }

            this.animFrameId = requestAnimationFrame(() => this._tick());
        }

        handleResize() {
            if (!this.container || !this.renderer || !this.camera) return;
            const rect = this.container.getBoundingClientRect();
            const width = rect.width || 800;
            const height = rect.height || 600;

            this.camera.aspect = width / height;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(width, height);
            this.wakeSimulation();
        }

        onWindowResize() {
            this.handleResize();
        }
    }

    // Export to global scope
    global.SageArtifactGraph = SageArtifactGraph;

})(window);
