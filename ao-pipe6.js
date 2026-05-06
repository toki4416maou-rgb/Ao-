// ═══════════════════════════════════════════════════════════════════════
// PIPE 6 (統合): ConceptGraph・12軸 + クオリア力場 → 意思決定野 → 因果推論野 → 出力生成
//
// 役割：
//   ① クオリア力場（PhenomenologicalField）の状態を
//      因果推論野（CIR）に渡す（動機づけ → 因果推論）
//   ② CIRが出力トークンを分解・設計する逆方向推論を追加
//   ③ PrefrontalDecisionCoreV1_1が「何を出力するか」を決定
//   ④ spatialベクトル合成をGPU（WebGL2）で実行
//      → ImageGenerator・AudioGeneratorに渡す
//
// GPU駆動：
//   spatialベクトルの重み付き合成 → WebGL2コンピュートシェーダー
//   CPUフォールバックあり
//
// ═══════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────
// GPU: spatialベクトル合成シェーダー
// 複数のspatialベクトルを重み付きで合成する
// input: Float32Array × N本 + 重みベクトル
// output: 合成されたFloat32Array
// ─────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────
// 統計トークナイザで意味トークンを抽出（ハードコードなし）
// PIPE4の4軸信用値を使って助詞・区切りを統計で除去する
// statTokが未学習の場合はスペース分割にフォールバック
// ─────────────────────────────────────────────────────────────────────
function _extractMeaningTokens(text, being) {
    try {
        const statTok = being && (
            being.statisticalTokenizer
            || (being.languageOutputDL
                && being.languageOutputDL.languageAcquisition
                && being.languageOutputDL.languageAcquisition.perceptualParser)
        );

        if (statTok && statTok._segment && statTok.tokenScores.size > 10) {
            // PIPE4が育っていれば統計で分割
            const segs = statTok._segment(text);
            return segs
                .filter(tok => {
                    const info = statTok.tokenScores.get(tok.surface);
                    if (!info) return tok.surface.length > 0;
                    // delimConf高い（区切り）またはsuffixConf高い（助詞）は除外
                    const isDelim  = (info.delimConf  || 0) > 0.5;
                    const isSuffix = (info.suffixConf || 0) > 0.6;
                    return !isDelim && !isSuffix && tok.surface.length > 0;
                })
                .map(tok => tok.surface)
                .filter(s => s.trim().length > 0);
        }
    } catch(e) {
        console.warn('[PIPE6] _extractMeaningTokens error:', e);
    }
    // フォールバック：スペースで分割（少なくともハードコード助詞よりはマシ）
    return text.split(/\s+/).filter(t => t.length > 0);
}

const FS_SPATIAL_BLEND = `#version 300 es
precision highp float;
uniform sampler2D u_vecs;   // N本のspatialベクトルをテクスチャに詰めたもの
uniform sampler2D u_weights; // 重みベクトル
uniform int u_count;         // ベクトル本数
uniform int u_dim;           // ベクトル次元（2104）
in vec2 v_uv;
out vec4 o;
void main() {
    int dim_i = int(gl_FragCoord.x); // 0~dim-1
    if (dim_i >= u_dim) { o = vec4(0.0); return; }
    float sum = 0.0;
    float wsum = 0.0;
    for (int k = 0; k < u_count; k++) {
        float uv_x = (float(dim_i) + 0.5) / float(u_dim);
        float uv_y = (float(k) + 0.5) / float(u_count);
        float val = texture(u_vecs, vec2(uv_x, uv_y)).r;
        float w   = texture(u_weights, vec2((float(k)+0.5)/float(u_count), 0.5)).r;
        sum  += val * w;
        wsum += w;
    }
    o = vec4(wsum > 0.0 ? sum / wsum : 0.0, 0.0, 0.0, 1.0);
}`;

// ─────────────────────────────────────────────────────────────────────
// GPUSpatialBlender: spatialベクトルのGPU合成器
// ─────────────────────────────────────────────────────────────────────
class GPUSpatialBlender {
    constructor() {
        this._gl      = null;
        this._program = null;
        this._ready   = false;
        this._init();
    }

    _init() {
        try {
            const canvas = document.createElement('canvas');
            canvas.width  = 2104; // spatial次元数
            canvas.height = 1;
            const gl = canvas.getContext('webgl2');
            if (!gl) { console.warn('[PIPE6/GPU] WebGL2非対応 → CPUフォールバック'); return; }

            // 頂点シェーダー（フルスクリーンクワッド）
            const VS = `#version 300 es
in vec2 a_pos; out vec2 v_uv;
void main() { gl_Position = vec4(a_pos,0,1); v_uv = a_pos*0.5+0.5; }`;

            const vs = this._compile(gl, gl.VERTEX_SHADER, VS);
            const fs = this._compile(gl, gl.FRAGMENT_SHADER, FS_SPATIAL_BLEND);
            if (!vs || !fs) return;

            const prog = gl.createProgram();
            gl.attachShader(prog, vs);
            gl.attachShader(prog, fs);
            gl.linkProgram(prog);
            if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
                console.warn('[PIPE6/GPU] シェーダーリンク失敗:', gl.getProgramInfoLog(prog));
                return;
            }

            // フルスクリーンクワッド
            const buf = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, buf);
            gl.bufferData(gl.ARRAY_BUFFER,
                new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);

            this._gl      = gl;
            this._program = prog;
            this._canvas  = canvas;
            this._buf     = buf;
            this._ready   = true;
            console.log('[PIPE6/GPU] spatialブレンダー初期化完了');
        } catch(e) {
            console.warn('[PIPE6/GPU] 初期化エラー:', e);
        }
    }

    _compile(gl, type, src) {
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
            console.warn('[PIPE6/GPU] シェーダーコンパイル失敗:', gl.getShaderInfoLog(s));
            return null;
        }
        return s;
    }

    // vectors: Float32Array[] (各2104次元)
    // weights: number[] (各ベクトルの重み)
    // return: Float32Array (合成結果)
    blend(vectors, weights) {
        if (!this._ready || vectors.length === 0) {
            return this._cpuBlend(vectors, weights);
        }
        try {
            const gl    = this._gl;
            const dim   = 2104;
            const count = vectors.length;

            gl.useProgram(this._program);
            gl.viewport(0, 0, dim, 1);

            // ベクトルテクスチャ（N × dim）
            const vecData = new Float32Array(dim * count);
            for (let k = 0; k < count; k++) {
                const v = vectors[k];
                for (let i = 0; i < Math.min(dim, v.length); i++) {
                    vecData[k * dim + i] = v[i];
                }
            }
            const texVecs = this._makeTex(gl, dim, count, vecData);

            // 重みテクスチャ（1 × count）
            const wData = new Float32Array(count);
            for (let k = 0; k < count; k++) wData[k] = weights[k] || 0;
            const texWeights = this._makeTex(gl, count, 1, wData);

            // ユニフォーム
            gl.uniform1i(gl.getUniformLocation(this._program, 'u_vecs'),    0);
            gl.uniform1i(gl.getUniformLocation(this._program, 'u_weights'), 1);
            gl.uniform1i(gl.getUniformLocation(this._program, 'u_count'),   count);
            gl.uniform1i(gl.getUniformLocation(this._program, 'u_dim'),     dim);

            gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texVecs);
            gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, texWeights);

            // 描画
            const posLoc = gl.getAttribLocation(this._program, 'a_pos');
            gl.bindBuffer(gl.ARRAY_BUFFER, this._buf);
            gl.enableVertexAttribArray(posLoc);
            gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            // 結果読み出し
            const result = new Float32Array(dim);
            gl.readPixels(0, 0, dim, 1, gl.RED, gl.FLOAT, result);

            gl.deleteTexture(texVecs);
            gl.deleteTexture(texWeights);

            return result;
        } catch(e) {
            console.warn('[PIPE6/GPU] ブレンドエラー → CPUフォールバック:', e);
            return this._cpuBlend(vectors, weights);
        }
    }

    _makeTex(gl, w, h, data) {
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, w, h, 0, gl.RED, gl.FLOAT, data);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        return tex;
    }

    // CPUフォールバック
    _cpuBlend(vectors, weights) {
        const dim = 2104;
        const result = new Float32Array(dim);
        let wsum = 0;
        for (let k = 0; k < vectors.length; k++) wsum += (weights[k] || 0);
        if (wsum === 0) return result;
        for (let k = 0; k < vectors.length; k++) {
            const w = (weights[k] || 0) / wsum;
            const v = vectors[k];
            for (let i = 0; i < Math.min(dim, v.length); i++) result[i] += v[i] * w;
        }
        return result;
    }
}

// ─────────────────────────────────────────────────────────────────────
// PIPE 6 メイン
// ─────────────────────────────────────────────────────────────────────
function attachPipe6(being) {
    if (!being) return;
    if (being._pipe6Attached) return;
    being._pipe6Attached = true;

    const qualiaField  = being.qualiaField;
    const cir          = being.causalInterventionReasoner;
    const prefrontal   = being.prefrontalCoreV1_1;
    const imageGen     = being.imageGenerator;
    const audioGen     = being.audioGenerator;
    const conceptGraph = being.conceptGraph || window._aoConceptGraph;

    if (!qualiaField || !cir || !prefrontal || !imageGen || !audioGen) {
        console.warn('[PIPE6] 依存未接続 - リトライ');
        setTimeout(() => { being._pipe6Attached = false; attachPipe6(being); }, 2000);
        return;
    }

    // GPU合成器を初期化
    const gpuBlender = new GPUSpatialBlender();
    being._gpuBlender = gpuBlender;

    // ─────────────────────────────────────────────────────────────────
    // ① クオリア力場 → CIR パス
    // placeState()のたびに力場状態をCIRに記録する
    // ─────────────────────────────────────────────────────────────────
    const origPlaceState = qualiaField.placeState.bind(qualiaField);
    qualiaField.placeState = function(id, state) {
        const pos = origPlaceState(id, state);
        try {
            // 力場の位置ベクトルを因果推論野に渡す
            // x軸：価値（正=快・負=不快）
            // y軸：覚醒度（高=活性・低=静止）
            const constraints = qualiaField.getActionConstraints();
            if (constraints.canAct && constraints.strength > 0.2) {
                cir.record(
                    `クオリア力場[${constraints.originMode}]`,
                    { fieldActive: false, strength: 0 },
                    {
                        fieldActive:   true,
                        strength:      constraints.strength,
                        direction:     constraints.direction,
                        originMode:    constraints.originMode,
                        globalActivity: constraints.globalActivity,
                        relationType:  'qualia-motivation',
                        grammarConf:   constraints.strength,
                        subject:       id,
                        predicate:     constraints.originMode,
                    }
                );
            }
        } catch(e) { console.warn('[PIPE6①] qualia→CIR error:', e); }
        return pos;
    };

    // ─────────────────────────────────────────────────────────────────
    // ② CIR に逆方向推論（出力トークン分解）を追加
    // 「角の吠えた犬」→ 概念トークンに分解 → spatialベクトル合成指示
    // ─────────────────────────────────────────────────────────────────
    cir.designOutput = function(outputRequest) {
        // outputRequest: { text, modality, mood }
        const { text, modality, mood } = outputRequest;
        if (!text || !modality) return null;

        try {
            // CIR履歴から関連するis-a/has-propertyパターンを検索
            const relatedRecords = this.history.filter(e =>
                e.stateAfter && (
                    e.stateAfter.relationType === 'is-a' ||
                    e.stateAfter.relationType === 'has-property'
                ) && text.includes(e.stateAfter.subject || '')
            );

            // 概念トークンに分解（簡易：空白・助詞で分割）
            const tokens = _extractMeaningTokens(text, being);

            // 各トークンのspatialベクトルを取得
            const spatialPairs = [];
            const audioPairs   = [];

            for (const token of tokens) {
                // VisualHypothesisTableから取得
                const imageAdapter = being.imageAdapter;
                if (imageAdapter && imageAdapter.hypothesisTable) {
                    const hyp = imageAdapter.hypothesisTable.hypotheses &&
                                imageAdapter.hypothesisTable.hypotheses.get(token);
                    if (hyp && hyp.spatial && hyp.spatial.samples >= 4) {
                        const n    = hyp.spatial.samples;
                        const mean = new Float32Array(hyp.spatial.sum.map(v => v / n));
                        // 信用値を重みとして使う
                        spatialPairs.push({ token, vector: mean, weight: hyp.confidence || 0.5 });
                    }
                }

                // AcousticHypothesisTableから取得
                const audioHyp = audioGen.hypothesisTable &&
                                 audioGen.hypothesisTable._entries &&
                                 audioGen.hypothesisTable._entries.get(token);
                if (audioHyp && audioHyp.count >= 2) {
                    audioPairs.push({
                        token,
                        avgEntropy:    audioHyp.sumEntropy    / audioHyp.count,
                        avgPMI:        audioHyp.sumPMI        / audioHyp.count,
                        avgTempo:      audioHyp.sumTempo      / audioHyp.count,
                        avgBrightness: audioHyp.sumBrightness / audioHyp.count,
                        weight:        Math.min(1, audioHyp.count / 10),
                    });
                }

                // ConceptGraphから親カテゴリのspatialも取得
                if (conceptGraph) {
                    const parents = conceptGraph.getParents(token);
                    for (const parent of parents) {
                        const parentHyp = imageAdapter &&
                            imageAdapter.hypothesisTable &&
                            imageAdapter.hypothesisTable.hypotheses &&
                            imageAdapter.hypothesisTable.hypotheses.get(parent);
                        if (parentHyp && parentHyp.spatial && parentHyp.spatial.samples >= 4) {
                            const n    = parentHyp.spatial.samples;
                            const mean = new Float32Array(parentHyp.spatial.sum.map(v => v / n));
                            // 親カテゴリは重みを下げる
                            spatialPairs.push({ token: parent, vector: mean, weight: (parentHyp.confidence || 0.5) * 0.6 });
                        }
                    }
                }
            }

            // CIRにこの設計を記録（学習素材）
            this.record(
                `出力設計[${modality}]:${text}`,
                { tokens: [], spatialPairs: 0 },
                {
                    tokens,
                    spatialCount:  spatialPairs.length,
                    audioCount:    audioPairs.length,
                    relationType:  'output-design',
                    grammarConf:   spatialPairs.length > 0 ? 0.8 : 0.3,
                    subject:       text,
                    predicate:     modality,
                }
            );

            return { tokens, spatialPairs, audioPairs, modality, text };

        } catch(e) {
            console.warn('[PIPE6②] designOutput error:', e);
            return null;
        }
    };

    // ─────────────────────────────────────────────────────────────────
    // ③ PrefrontalDecisionCore → 出力生成の自律トリガー
    // 意思決定野が「出力する」と決めたとき自律的に生成を走らせる
    // ─────────────────────────────────────────────────────────────────
    const origDecide = prefrontal.decide && prefrontal.decide.bind(prefrontal);
    if (origDecide) {
        prefrontal.decide = function(input) {
            const result = origDecide(input);
            try {
                // 意思決定結果に「画像/音声生成」意図が含まれるか
                const chosen = result && (result.chosen || result.decision);
                if (chosen && chosen.text) {
                    const text = chosen.text;
                    const isImageRequest = /画像|描|生成|イメージ|見せ/.test(text);
                    const isAudioRequest = /音|サウンド|音楽|鳴らし/.test(text);

                    if ((isImageRequest || isAudioRequest) && cir.designOutput) {
                        const modality = isImageRequest ? 'image' : 'audio';
                        const design = cir.designOutput({
                            text,
                            modality,
                            mood: being.state || {},
                        });

                        if (design && design.spatialPairs.length > 0 && modality === 'image') {
                            // GPU合成してImageGeneratorに渡す
                            _triggerImageFromDesign(being, design, gpuBlender);
                        }
                        if (design && design.audioPairs.length > 0 && modality === 'audio') {
                            _triggerAudioFromDesign(being, design);
                        }
                    }
                }
            } catch(e) { console.warn('[PIPE6③] decide hook error:', e); }
            return result;
        };
    }

    // ─────────────────────────────────────────────────────────────────
    // ④ 外部から「〜を画像にして」と呼べるAPI
    // being.generateFromText('角の吠えた犬', 'image') で呼べる
    // ─────────────────────────────────────────────────────────────────
    being.generateFromText = async function(text, modality = 'image') {
        if (!cir.designOutput) return null;

        being.addLog && being.addLog(`[PIPE6] generateFromText: "${text}" → ${modality}`);

        const design = cir.designOutput({ text, modality, mood: being.state || {} });
        if (!design) return null;

        if (modality === 'image') {
            return _triggerImageFromDesign(being, design, gpuBlender);
        } else if (modality === 'audio') {
            return _triggerAudioFromDesign(being, design);
        }
        return null;
    };


    // ═══════════════════════════════════════════════════════════════════
    // 統合: ConceptGraph・12軸 → 画像・音声生成器（旧PIPE5）
    // ═══════════════════════════════════════════════════════════════════

    // ConceptGraph + 12軸 → conceptHints を生成する関数
    function extractConceptGraphHints(queryLabel) {
        const hints = [];
        if (!conceptGraph) return hints;
        const targets = queryLabel
            ? [queryLabel, ...[...conceptGraph.getParents(queryLabel)]]
            : [...conceptGraph.groups.keys()];
        for (const category of targets) {
            const members = conceptGraph.groups.get(category);
            if (!members || members.size === 0) continue;
            const props = conceptGraph.inferCategoryProperties(category);
            const axisScore = worldView
                ? ((worldView.getAxis('hierarchy')   || 0) * 0.4
                +  (worldView.getAxis('causality')   || 0) * 0.4
                +  (worldView.getAxis('information') || 0) * 0.2)
                : 0.5;
            hints.push({
                label: category, members: [...members], memberCount: members.size,
                properties: props, axisScore,
                spatialWeight:   Math.min(1.0, members.size / 5),
                complexityScore: Math.min(1.0, props.length / 5),
            });
        }
        hints.sort((a, b) => (b.axisScore * b.memberCount) - (a.axisScore * a.memberCount));
        return hints.slice(0, 5);
    }

    // ImageGenerator.generateFromMood() をラップ（ConceptGraphヒント追加）
    const origImageGen = imageGen.generateFromMood.bind(imageGen);
    imageGen.generateFromMood = async function(mood, conceptHints) {
        try {
            const cgHints = extractConceptGraphHints();
            if (cgHints.length > 0) {
                const merged = [...(conceptHints || [])];
                for (const cg of cgHints) {
                    if (!merged.find(h => h.label === cg.label)) {
                        merged.push({
                            label: cg.label, count: cg.memberCount,
                            avgBrightness:      0.3 + cg.spatialWeight * 0.5,
                            brightnessReliability: cg.axisScore,
                            hueReliability:     cg.axisScore * 0.8,
                            shapeReliability:   cg.axisScore,
                            textureReliability: cg.complexityScore,
                            dominantHue:        _categoryToHue(cg.label),
                            dominantHueName:    _categoryToHueName(cg.label),
                            meanVector: null, fromConceptGraph: true,
                            properties: cg.properties,
                        });
                    }
                }
                being.addLog && being.addLog(
                    `[PIPE6] 画像: ConceptGraph[${cgHints.map(h=>h.label).join(',')}] → hints追加`
                );
                return origImageGen(mood, merged);
            }
        } catch(e) { console.warn('[PIPE6] imageGen hook error:', e); }
        return origImageGen(mood, conceptHints);
    };

    // AudioGenerator.generateFromMood() をラップ（ConceptGraphヒント追加）
    const origAudioGen = audioGen.generateFromMood.bind(audioGen);
    audioGen.generateFromMood = async function(mood, conceptHints) {
        try {
            const cgHints = extractConceptGraphHints();
            if (cgHints.length > 0) {
                const merged = [...(conceptHints || [])];
                for (const cg of cgHints) {
                    if (!merged.find(h => h.label === cg.label)) {
                        merged.push({
                            label: cg.label, count: cg.memberCount,
                            avgEntropy:    cg.complexityScore,
                            avgPMI:        cg.axisScore,
                            avgTempo:      1.0 - cg.spatialWeight * 0.5,
                            avgBrightness: 0.3 + cg.axisScore * 0.5,
                            fromConceptGraph: true,
                        });
                    }
                }
                being.addLog && being.addLog(
                    `[PIPE6] 音声: ConceptGraph[${cgHints.map(h=>h.label).join(',')}] → hints追加`
                );
                return origAudioGen(mood, merged);
            }
        } catch(e) { console.warn('[PIPE6] audioGen hook error:', e); }
        return origAudioGen(mood, conceptHints);
    };

    // ConceptGraph更新時に生成器へ通知
    const origAddRelP6 = conceptGraph.addRelation.bind(conceptGraph);
    conceptGraph.addRelation = function(subject, relation, object) {
        origAddRelP6(subject, relation, object);
        being.addLog && being.addLog(
            `[PIPE6] 概念更新: ${subject} ${relation} ${object} → 次回生成に反映`
        );
    };


    // ═══════════════════════════════════════════════════════════════════
    // 動画生成パイプ：
    // CIR.designVideo() → シーン設計 → GPU並列フレーム生成 → 統合
    // VideoPerceptualParserを逆利用してトークン→フレームに変換
    // ═══════════════════════════════════════════════════════════════════

    // CIRに動画設計メソッドを追加
    // 「角の犬が走る」→ シーン列（因果推論で全体構成を決める）
    cir.designVideo = function(request) {
        const { text, duration = 6 } = request;
        if (!text) return null;

        try {
            // テキストを動作トークンに分解
            const motionWords = _extractMeaningTokens(text, being);

            // CIR履歴から動作の因果パターンを検索
            // 「走る→ジャンプ」という因果関係が学習済みなら使う
            const causalChain = [];
            for (let i = 0; i < motionWords.length; i++) {
                const word = motionWords[i];

                // ConceptGraphから空間パターンを取得
                const imageAdapter = being && being.imageAdapter;
                const hyp = imageAdapter && imageAdapter.hypothesisTable &&
                            imageAdapter.hypothesisTable.hypotheses &&
                            imageAdapter.hypothesisTable.hypotheses.get(word);

                // VideoPerceptualParserのクラスタ情報から動作クラスタを取得
                const videoParser = being && being.videoAdapter &&
                                   being.videoAdapter.videoParser;
                const clusterId = videoParser
                    ? videoParser._quantize(
                        hyp && hyp.spatial ? hyp.spatial.sum.map(v => v / Math.max(1, hyp.spatial.samples)) : []
                    ) : i % 16;

                // シーンとして登録
                causalChain.push({
                    label:       word,
                    clusterId,
                    spatial:     hyp && hyp.spatial && hyp.spatial.samples >= 4
                        ? new Float32Array(hyp.spatial.sum.map(v => v / hyp.spatial.samples))
                        : null,
                    confidence:  hyp ? (hyp.confidence || 0.3) : 0.2,
                    // このシーンの持続フレーム数（全体の均等割り）
                    frameCount:  Math.ceil((duration * 6) / motionWords.length),
                    // 前シーンからの遷移（因果推論で決める）
                    transition:  i === 0 ? 'start' : 'blend',
                });
            }

            // CIRに動画設計を記録（学習素材）
            this.record(
                `動画設計:${text}`,
                { scenes: 0 },
                {
                    scenes:       causalChain.length,
                    totalFrames:  duration * 6,
                    relationType: 'video-design',
                    grammarConf:  0.8,
                    subject:      text,
                    predicate:    'video',
                }
            );

            being.addLog && being.addLog(
                `[PIPE6/Video] 動画設計完了: ${causalChain.map(s=>s.label).join('→')} (${duration}秒)`
            );

            return { text, scenes: causalChain, duration, fps: 6 };

        } catch(e) {
            console.warn('[PIPE6/Video] designVideo error:', e);
            return null;
        }
    };

    // VideoGenerator.generateFromMood() をラップ
    // ConceptGraphヒントを動画にも追加
    const videoGen = being.videoGenerator;
    if (videoGen) {
        const origVideoGen = videoGen.generateFromMood.bind(videoGen);
        videoGen.generateFromMood = async function(mood, duration, conceptHints) {
            try {
                const cgHints = extractConceptGraphHints();
                if (cgHints.length > 0) {
                    const merged = [...(conceptHints || [])];
                    for (const cg of cgHints) {
                        if (!merged.find(h => h.label === cg.label)) {
                            merged.push({
                                label: cg.label, count: cg.memberCount,
                                avgBrightness:         0.3 + cg.spatialWeight * 0.5,
                                brightnessReliability: cg.axisScore,
                                hueReliability:        cg.axisScore * 0.8,
                                shapeReliability:      cg.axisScore,
                                textureReliability:    cg.complexityScore,
                                meanVector:            null,
                                fromConceptGraph:      true,
                            });
                        }
                    }
                    return origVideoGen(mood, duration, merged);
                }
            } catch(e) { console.warn('[PIPE6/Video] videoGen hook error:', e); }
            return origVideoGen(mood, duration, conceptHints);
        };

        // テキストから動画を生成するAPI
        being.generateVideoFromText = async function(text, duration = 6) {
            if (!cir.designVideo) return null;

            being.addLog && being.addLog(`[PIPE6/Video] generateVideoFromText: "${text}"`);

            const design = cir.designVideo({ text, duration });
            if (!design || design.scenes.length === 0) return null;

            return _triggerVideoFromDesign(being, design, gpuBlender);
        };
    }


    // ═══════════════════════════════════════════════════════════════════
    // 編集統合パイプ：
    // CIR.designVideo() → モードA（EditingCognitionLayer）
    //                  → モードB（generateVideoFromText）
    // TaskManagerと連携して進捗を追跡
    // ═══════════════════════════════════════════════════════════════════

    const editorManager  = being.editorIntegrationManager;
    const videoEditing   = being.videoEditing;
    const taskMgr        = being.taskManager;

    if (editorManager || videoEditing) {

        // ── EditorIntegrationManager.processTask() をラップ ────────────
        // モードA/B共通でCIR.designVideo()を通してから処理する
        if (editorManager && editorManager.processTask) {
            const origProcessTask = editorManager.processTask.bind(editorManager);
            editorManager.processTask = async function(editorType, task, assets) {
                try {
                    being.addLog && being.addLog(
                        `[PIPE6/Edit] processTask開始: mode=${this.currentMode} task="${task}"`
                    );

                    // CIRで動画構成を設計（モードA・B共通）
                    const design = cir.designVideo({ text: task, duration: 30 });

                    if (taskMgr) {
                        taskMgr.createTask('edit-design', 'edit', `編集設計: ${task}`);
                        taskMgr.updateProgress('edit-design', 0.2);
                    }

                    if (this.currentMode === 'simple') {
                        // ── モードB：CIR設計 → generateVideoFromText ──
                        taskMgr && taskMgr.updateProgress('edit-design', 0.4);

                        const videoResult = being.generateVideoFromText
                            ? await being.generateVideoFromText(task, 30)
                            : null;

                        taskMgr && taskMgr.updateProgress('edit-design', 1.0);
                        taskMgr && taskMgr.completeTask('edit-design');

                        if (videoResult) {
                            being.addLog && being.addLog(
                                `[PIPE6/Edit] モードB生成完了: ${videoResult.frames.length}フレーム`
                            );
                            return {
                                success:  true,
                                mode:     'simple',
                                design,
                                frames:   videoResult.frames,
                                fps:      videoResult.fps,
                                duration: videoResult.duration,
                                description: videoResult.description,
                            };
                        }
                        // videoGeneratorが使えなければ既存処理にフォールバック
                        return origProcessTask(editorType, task, assets);

                    } else {
                        // ── モードA：CIR設計 → EditingCognitionLayerに渡す ──
                        if (videoEditing && videoEditing.cognition) {
                            // CIRの設計結果をEditingCognitionLayerに注入
                            videoEditing.cognition.setIntent(task);

                            // CIRが設計したシーン列をoperationSequenceに変換
                            if (design && design.scenes) {
                                const cirOps = design.scenes.map((scene, i) => ({
                                    type:        `scene_${i}`,
                                    description: scene.label,
                                    clusterId:   scene.clusterId,
                                    confidence:  scene.confidence,
                                    frameCount:  scene.frameCount,
                                    transition:  scene.transition,
                                    completed:   false,
                                    // CIR由来フラグ
                                    fromCIR:     true,
                                }));

                                // 既存のoperationSequenceの前にCIR設計を挿入
                                videoEditing.cognition.operationSequence = [
                                    ...cirOps,
                                    ...videoEditing.cognition.operationSequence,
                                ];

                                being.addLog && being.addLog(
                                    `[PIPE6/Edit] モードA: CIR設計${cirOps.length}シーン → EditingCognitionLayer注入`
                                );
                            }
                        }

                        taskMgr && taskMgr.updateProgress('edit-design', 0.5);
                        const result = await origProcessTask(editorType, task, assets);
                        taskMgr && taskMgr.updateProgress('edit-design', 1.0);
                        taskMgr && taskMgr.completeTask('edit-design');

                        return result;
                    }

                } catch(e) {
                    console.warn('[PIPE6/Edit] processTask hook error:', e);
                    taskMgr && taskMgr.completeTask('edit-design');
                    return origProcessTask(editorType, task, assets);
                }
            };
        }

        // ── VideoEditingIntegration.createProject() をラップ ──────────
        // プロジェクト作成時にCIRの設計を注入する
        if (videoEditing && videoEditing.createProject) {
            const origCreateProject = videoEditing.createProject.bind(videoEditing);
            videoEditing.createProject = async function(intent, folderPath, projectName, format) {
                try {
                    // CIRで意図を因果推論
                    const design = cir.designVideo({ text: intent, duration: 30 });

                    if (design && design.scenes && this.timelineBuilder) {
                        // CIRのシーン設計をTimelineBuilderに事前注入
                        // シーンの順序・遷移を因果推論で決めた構成にする
                        being.addLog && being.addLog(
                            `[PIPE6/Edit] createProject: CIR設計${design.scenes.length}シーン → TimelineBuilder注入`
                        );

                        // TimelineBuilderにCIR由来のシーンメタを設定
                        this.timelineBuilder._cirDesign = design.scenes.map(s => ({
                            label:      s.label,
                            duration:   s.frameCount / 6, // フレーム数→秒
                            transition: s.transition,
                            confidence: s.confidence,
                        }));
                    }

                    taskMgr && taskMgr.createTask('create-project', 'edit', `プロジェクト作成: ${projectName}`);
                    const result = await origCreateProject(intent, folderPath, projectName, format);
                    taskMgr && taskMgr.completeTask('create-project');

                    if (result.success) {
                        // CIRにプロジェクト作成を記録（学習素材）
                        cir.record(
                            `プロジェクト作成[${projectName}]`,
                            { hasProject: false },
                            {
                                hasProject:   true,
                                intent,
                                format,
                                sceneCount:   design ? design.scenes.length : 0,
                                relationType: 'project-created',
                                grammarConf:  0.8,
                                subject:      intent,
                                predicate:    projectName,
                            }
                        );
                    }

                    return result;
                } catch(e) {
                    console.warn('[PIPE6/Edit] createProject hook error:', e);
                    return origCreateProject(intent, folderPath, projectName, format);
                }
            };
        }

        being.addLog && being.addLog('[PIPE6/Edit] 編集統合パイプ接続完了');
    }

    // ═══════════════════════════════════════════════════════════════════
    // 編集タブパイプ：
    // code / document / audio / slide / 実行
    // ActionPlanGenerator → CIR.designOutput()に置換
    // 各Adapter → ConceptGraph・12軸・AudioGenerator と接続
    // ═══════════════════════════════════════════════════════════════════

    const actionPlanGen = editorManager && editorManager.actionPlanGenerator;
    const execLayer     = editorManager && editorManager.executionLayer;

    if (actionPlanGen) {

        // ── ActionPlanGenerator.generatePlan() をラップ ────────────────
        // キーワードマッチをCIR.designOutput()に置換
        const origGeneratePlan = actionPlanGen.generatePlan.bind(actionPlanGen);
        actionPlanGen.generatePlan = function(editorType, task, assets) {
            try {
                // CIRで意図を因果推論して設計
                const design = cir.designOutput({ text: task, modality: editorType });

                if (design && design.tokens && design.tokens.length > 0) {
                    being.addLog && being.addLog(
                        `[PIPE6/Plan] CIR設計: ${editorType} → tokens=[${design.tokens.join(',')}]`
                    );

                    const plan = {
                        id:         `plan_${Date.now()}`,
                        editorType,
                        task,
                        assets,
                        actions:    [],
                        fromCIR:    true,
                        cirDesign:  design,
                        timestamp:  Date.now(),
                    };

                    // タイプ別にCIR設計をアクションに変換
                    switch(editorType) {
                        case 'code':
                            plan.actions = _cirToCodeActions(design, task, assets);
                            break;
                        case 'document':
                            plan.actions = _cirToDocActions(design, task, assets);
                            break;
                        case 'audio':
                            plan.actions = _cirToAudioActions(design, task, assets, being);
                            break;
                        case 'slide':
                            plan.actions = _cirToSlideActions(design, task, assets, being);
                            break;
                        case 'video':
                            plan.actions = _cirToVideoActions(design, task, assets);
                            break;
                        default:
                            plan.actions = design.tokens.map(t => ({
                                type: 'generic', description: t, fromCIR: true
                            }));
                    }

                    // CIRに計画を記録（学習素材）
                    cir.record(
                        `編集計画[${editorType}]:${task}`,
                        { actionCount: 0 },
                        {
                            actionCount:  plan.actions.length,
                            editorType,
                            tokens:       design.tokens,
                            relationType: 'edit-plan',
                            grammarConf:  0.8,
                            subject:      task,
                            predicate:    editorType,
                        }
                    );

                    return plan;
                }
            } catch(e) {
                console.warn('[PIPE6/Plan] generatePlan hook error:', e);
            }
            // フォールバック：既存処理
            return origGeneratePlan(editorType, task, assets);
        };
    }

    // ── 実行ボタン（executeEditorTask）にTaskManager連携を追加 ─────
    // window.executeEditorTask をラップしてCIRとTaskManagerを通す
    if (window.executeEditorTask) {
        const origExecute = window.executeEditorTask;
        window.executeEditorTask = async function() {
            taskMgr && taskMgr.createTask('editor-exec', 'edit', '編集タスク実行中');
            taskMgr && taskMgr.updateFocus(0.9);
            try {
                const result = await origExecute();
                taskMgr && taskMgr.completeTask('editor-exec');
                taskMgr && taskMgr.updateFocus(0.5);
                return result;
            } catch(e) {
                taskMgr && taskMgr.completeTask('editor-exec');
                throw e;
            }
        };
    }

    being.addLog && being.addLog('[PIPE6/Editors] 全編集タブパイプ接続完了');




    // ═══════════════════════════════════════════════════════════════════
    // MREパイプ：
    // ① QualiaSnapshot → ConceptGraph saliency追加
    // ② MRE.renderImage() → spatialベクトル → ImageRenderer
    // ③ MRE.renderVideo() → CIR.designVideo() → GPU並列フレーム
    // ═══════════════════════════════════════════════════════════════════

    const mre = being.mre;
    if (mre) {

        // ── ① QualiaSnapshot.capture() をラップ ───────────────────────
        // ConceptGraphのカテゴリをsaliencyに追加する
        // MREが「今何を知っているか」を反映できるようになる
        const origCaptureProto = window.QualiaSnapshot &&
                                 window.QualiaSnapshot.prototype.capture;
        if (origCaptureProto) {
            window.QualiaSnapshot.prototype.capture = function() {
                const snapshot = origCaptureProto.call(this);
                try {
                    const cg = this.being.conceptGraph || window._aoConceptGraph;
                    if (cg && cg.groups.size > 0) {
                        // ConceptGraphのカテゴリをsaliencyとして追加
                        const cgSaliency = [...cg.groups.entries()]
                            .map(([category, members]) => ({
                                concept:      category,
                                depth:        members.size / 10,
                                saliency:     Math.min(1.0, members.size / 5 * 0.8),
                                fromConcept:  true,
                                memberCount:  members.size,
                            }))
                            .filter(c => c.saliency > 0.1)
                            .sort((a, b) => b.saliency - a.saliency)
                            .slice(0, 5);

                        // 既存saliencyとマージ（概念由来を後ろに）
                        snapshot.saliency = [
                            ...snapshot.saliency,
                            ...cgSaliency.filter(cg =>
                                !snapshot.saliency.find(s => s.concept === cg.concept)
                            ),
                        ].sort((a, b) => b.saliency - a.saliency);
                    }
                } catch(e) { console.warn('[PIPE6/MRE①] capture hook error:', e); }
                return snapshot;
            };
        }

        // ── ② MRE.renderImage() をラップ ─────────────────────────────
        // saliencyトップ概念のspatialベクトルをGPU合成してImageRendererに渡す
        const origRenderImage = mre.renderImage.bind(mre);
        mre.renderImage = function() {
            try {
                const intent   = mre._buildIntent('image');
                const snapshot = mre._captureQualia(intent);

                // saliencyトップ5概念のspatialベクトルを取得
                const spatialPairs = [];
                const weights      = [];
                for (const sal of snapshot.saliency.slice(0, 5)) {
                    const hyp = being.imageAdapter &&
                                being.imageAdapter.hypothesisTable &&
                                being.imageAdapter.hypothesisTable.hypotheses &&
                                being.imageAdapter.hypothesisTable.hypotheses.get(sal.concept);
                    if (hyp && hyp.spatial && hyp.spatial.samples >= 4) {
                        const n = hyp.spatial.samples;
                        spatialPairs.push(new Float32Array(hyp.spatial.sum.map(v => v / n)));
                        weights.push(sal.saliency);
                    }
                }

                if (spatialPairs.length > 0) {
                    // GPU合成
                    const blended = being._gpuBlender
                        ? being._gpuBlender.blend(spatialPairs, weights)
                        : null;

                    if (blended) {
                        // snapshotにspatialVectorを追加してImageRendererに渡す
                        snapshot.spatialVector = blended;
                        snapshot.attributes.fromSpatial = true;

                        being.addLog && being.addLog(
                            `[PIPE6/MRE②] 画像: ${spatialPairs.length}概念のspatialをGPU合成`
                        );

                        // ImageRendererのrender()でspatialVectorを使う
                        const dataUrl = mre.imgRenderer.render(snapshot);
                        const desc    = mre._describeOutput(snapshot, 'image');
                        being.addLog && being.addLog(`[MRE] 画像レンダリング完了(GPU): ${desc}`);
                        return { dataUrl, description: desc, intent, snapshot };
                    }
                }
            } catch(e) { console.warn('[PIPE6/MRE②] renderImage hook error:', e); }
            return origRenderImage();
        };

        // ── ③ MRE.renderVideo() をラップ ─────────────────────────────
        // saliencyトップ概念をCIR.designVideo()に渡して因果設計してから生成
        const origRenderVideo = mre.renderVideo.bind(mre);
        mre.renderVideo = async function(frameCount = 12, fps = 8) {
            try {
                const intent   = mre._buildIntent('video');
                const snapshot = mre._captureQualia(intent);

                // saliencyトップ概念でテキストを構成
                const topConcepts = snapshot.saliency
                    .slice(0, 3)
                    .map(s => s.concept)
                    .join(' ');

                if (topConcepts && cir.designVideo) {
                    being.addLog && being.addLog(
                        `[PIPE6/MRE③] 動画: CIR.designVideo("${topConcepts}")`
                    );

                    const duration = Math.ceil(frameCount / fps);
                    const design   = cir.designVideo({ text: topConcepts, duration });

                    if (design && design.scenes.length > 0 && being._gpuBlender) {
                        const result = await _triggerVideoFromDesign(being, design, being._gpuBlender);
                        if (result) {
                            being.addLog && being.addLog(
                                `[MRE] 動画レンダリング完了(CIR+GPU): ${result.frames.length}フレーム`
                            );
                            return {
                                frames:      result.frames,
                                fps,
                                description: result.description,
                                intent,
                                snapshot,
                            };
                        }
                    }
                }
            } catch(e) { console.warn('[PIPE6/MRE③] renderVideo hook error:', e); }
            return origRenderVideo(frameCount, fps);
        };

        being.addLog && being.addLog('[PIPE6/MRE] MRE → ConceptGraph・CIR・GPU パイプ接続完了');
    }


    // ═══════════════════════════════════════════════════════════════════
    // CSEパイプ：概念生成・ラベル付け機構 ↔ 因果推論野
    // ① CSE._validateAndStore() → ConceptGraph登録
    // ② CIR履歴 → CSE._pickAttrValue()のヒント
    // ③ CSE.synthesize()完了 → CIRに記録
    // ═══════════════════════════════════════════════════════════════════

    const cse = being.cse;
    if (cse) {

        // ── ① _validateAndStore() をラップ ────────────────────────────
        // 検証通過した概念をConceptGraphに自動登録する
        const origValidateStore = cse._validateAndStore.bind(cse);
        cse._validateAndStore = function(candidate, space) {
            const result = origValidateStore(candidate, space);
            if (!result) return result;

            try {
                const cg  = being.conceptGraph || window._aoConceptGraph;
                const label = result.primaryLabel || result.id;

                if (cg && label) {
                    // 合成元概念をis-aまたはhas-propertyとしてConceptGraphに登録
                    if (result.operation === 'merge' && result.sourceIds) {
                        // merge: 合成元 is-a 新概念
                        for (const srcId of result.sourceIds) {
                            cg.addRelation(srcId, 'is-a', label);
                        }
                    } else if (result.operation === 'add_attr' || result.operation === 'replace_attr') {
                        // 属性追加: 新概念 has-property 属性
                        const attrs = result.attributes || {};
                        for (const [k, v] of Object.entries(attrs)) {
                            cg.addRelation(label, 'has-property', `${k}:${v}`);
                        }
                    }

                    // ③ CIRに新概念生成を記録（因果推論の素材）
                    cir.record(
                        `概念生成[${result.operation}]:${label}`,
                        { conceptCount: cse.synthesized.length - 1 },
                        {
                            conceptCount:  cse.synthesized.length,
                            label,
                            operation:     result.operation,
                            validScore:    result.validationScore,
                            sourceIds:     result.sourceIds || [],
                            relationType:  'concept-synthesized',
                            grammarConf:   result.validationScore || 0.5,
                            subject:       label,
                            predicate:     result.operation,
                        }
                    );

                    being.addLog && being.addLog(
                        `[PIPE6/CSE] 新概念→ConceptGraph: 「${label}」op=${result.operation} score=${(result.validationScore||0).toFixed(2)}`
                    );
                }
            } catch(e) { console.warn('[PIPE6/CSE①] error:', e); }
            return result;
        };

        // ── ② _pickAttrValue() をラップ ───────────────────────────────
        // CIR履歴から高信用値の述語を属性値候補に追加する
        // ランダム選択ではなく因果推論が蓄積した語彙を優先使用
        const origPickAttr = cse._pickAttrValue.bind(cse);
        cse._pickAttrValue = function() {
            try {
                // CIR履歴から grammarConf > 0.6 の述語を収集
                const cirVocab = cir.history
                    .filter(e => e.stateAfter &&
                        (e.stateAfter.grammarConf || 0) > 0.6 &&
                        e.stateAfter.predicate)
                    .map(e => e.stateAfter.predicate)
                    .filter(Boolean);

                if (cirVocab.length > 0 && Math.random() > 0.4) {
                    // 40%の確率でCIR由来の語彙を使う（残り60%は既存処理）
                    const val = cirVocab[Math.floor(Math.random() * cirVocab.length)];
                    being.addLog && being.addLog(
                        `[PIPE6/CSE②] 属性値をCIR由来で選択: "${val}"`
                    );
                    return val;
                }
            } catch(e) { console.warn('[PIPE6/CSE②] error:', e); }
            return origPickAttr();
        };

        // ── cycle() 完了後にConceptGraph→12軸更新をトリガー ──────────
        const origCycle = cse.cycle.bind(cse);
        cse.cycle = function() {
            const generated = origCycle();
            try {
                if (generated && generated.length > 0 && being.worldView) {
                    // 新概念が生まれるたびにinformation軸を少し伸ばす
                    being.worldView.growAxis &&
                        being.worldView.growAxis('information', 0.01 * generated.length);
                    // 合成成功 → causality軸も成長
                    being.worldView.growAxis &&
                        being.worldView.growAxis('causality', 0.005 * generated.length);
                }
            } catch(e) { console.warn('[PIPE6/CSE cycle] error:', e); }
            return generated;
        };

        being.addLog && being.addLog('[PIPE6/CSE] 概念生成・ラベル付け機構 ↔ CIR パイプ接続完了');
    }


    // ═══════════════════════════════════════════════════════════════════
    // エピソード記憶パイプ：
    // ① episodicMemory.store() → CIR（体験を因果推論の素材に）
    // ② episodicMemory.retrieve() → 空間野（想起時にspatialも引っ張る）
    // ③ CIR.record() → episodicMemory（因果推論結果をエピソードとして記録）
    // ═══════════════════════════════════════════════════════════════════

    const episodic = being.episodicMemory || being.distributedEpisodic;

    if (episodic) {

        // ── ① store() をラップ → CIRに体験を渡す ──────────────────────
        const origStore = episodic.store.bind(episodic);
        episodic.store = function(fragment) {
            origStore(fragment);
            try {
                if (!fragment) return;

                const text = typeof fragment === 'string'
                    ? fragment
                    : (fragment.summary || fragment.text || fragment.content || JSON.stringify(fragment));

                if (!text) return;

                // テキストから意味トークンを抽出してCIRに渡す
                const tokens = _extractMeaningTokens(text, being);

                if (tokens.length > 0 && cir) {
                    cir.record(
                        `エピソード記憶:${tokens.slice(0, 3).join('+')}`,
                        { episodeCount: episodic.episodes.length - 1 },
                        {
                            episodeCount:  episodic.episodes.length,
                            tokens,
                            text:          text.slice(0, 100),
                            relationType:  'episodic-store',
                            grammarConf:   0.6,
                            subject:       tokens[0] || '体験',
                            predicate:     tokens[1] || '記録',
                        }
                    );

                    // ConceptGraphにも体験として登録
                    const cg = being.conceptGraph || window._aoConceptGraph;
                    if (cg && tokens.length >= 2) {
                        cg.addRelation(tokens[0], 'experienced', tokens[1]);
                    }

                    being.addLog && being.addLog(
                        `[PIPE6/Episodic①] store→CIR: [${tokens.slice(0,3).join(',')}]`
                    );
                }
            } catch(e) { console.warn('[PIPE6/Episodic①] error:', e); }
        };

        // ── ② retrieve() をラップ → 空間野のspatialも一緒に返す ───────
        const origRetrieve = episodic.retrieve.bind(episodic);
        episodic.retrieve = function(query) {
            const episodes = origRetrieve(query);
            try {
                // 各エピソードに空間野のspatialベクトルを付与
                for (const ep of episodes) {
                    const text = typeof ep.fragment === 'string'
                        ? ep.fragment
                        : (ep.fragment && (ep.fragment.summary || ep.fragment.text)) || '';

                    const tokens = _extractMeaningTokens(text, being);

                    // spatialベクトルを取得
                    const spatialPairs = [];
                    for (const tok of tokens) {
                        const hyp = being.imageAdapter &&
                            being.imageAdapter.hypothesisTable &&
                            being.imageAdapter.hypothesisTable.hypotheses &&
                            being.imageAdapter.hypothesisTable.hypotheses.get(tok);
                        if (hyp && hyp.spatial && hyp.spatial.samples >= 4) {
                            const n = hyp.spatial.samples;
                            spatialPairs.push({
                                token:  tok,
                                vector: new Float32Array(hyp.spatial.sum.map(v => v / n)),
                                conf:   hyp.confidence || 0.5,
                            });
                        }
                    }

                    // CIR履歴から関連する因果パターンを取得
                    const causalPatterns = cir ? cir.history
                        .filter(e => tokens.some(t =>
                            e.action && e.action.includes(t)))
                        .slice(-3) : [];

                    // エピソードにspatial・因果パターンを付与
                    ep._spatial      = spatialPairs;
                    ep._causal       = causalPatterns;
                    ep._tokens       = tokens;
                }

                if (episodes.length > 0) {
                    being.addLog && being.addLog(
                        `[PIPE6/Episodic②] retrieve: ${episodes.length}件 spatial付与`
                    );
                }
            } catch(e) { console.warn('[PIPE6/Episodic②] error:', e); }
            return episodes;
        };

        // ── ③ CIR.record() に → episodicMemoryへの記録を追加 ──────────
        // 高信用値の因果推論結果をエピソードとして保存する
        const origCirRecord = cir.record.bind(cir);
        cir.record = function(action, stateBefore, stateAfter) {
            origCirRecord(action, stateBefore, stateAfter);
            try {
                // grammarConf > 0.7 の高信用値パターンだけエピソードに記録
                const conf = stateAfter && (stateAfter.grammarConf || 0);
                if (conf > 0.7 && stateAfter.subject && stateAfter.predicate) {
                    const fragment = {
                        summary:  `${stateAfter.subject}→${stateAfter.predicate}(${stateAfter.relationType})`,
                        action,
                        conf,
                        fromCIR:  true,
                        timestamp: Date.now(),
                    };
                    // store()のラップを通さず直接保存（無限ループ防止）
                    origStore(fragment);
                }
            } catch(e) { console.warn('[PIPE6/Episodic③] error:', e); }
        };

        being.addLog && being.addLog('[PIPE6/Episodic] エピソード記憶 ↔ CIR・空間野 パイプ接続完了');
    }


    // ═══════════════════════════════════════════════════════════════════
    // 4軸学習連携パイプ：
    // PIPE4の4軸信用値（接尾語・語末・位置・区切り）を
    // ① TemplateSelector.record() のstateVectorに追加
    // ② LanguageInputDL.parse() の理解深度に反映
    // ③ episodicMemory.store() のエピソードに付与
    // ═══════════════════════════════════════════════════════════════════

    const statTok = being.statisticalTokenizer
        || (being.languageOutputDL
            && being.languageOutputDL.languageAcquisition
            && being.languageOutputDL.languageAcquisition.perceptualParser);

    const templateSel = being.templateSelector;
    const langDL      = being.languageInputDL;
    const episodic2   = being.episodicMemory || being.distributedEpisodic;

    if (statTok && statTok._axis4) {

        // ── ① TemplateSelector.record() をラップ ──────────────────────
        // stateVectorに4軸平均信用値を追加して
        // 「この言語状態でこのテンプレートが使われた」を精密に記録する
        if (templateSel) {
            const origRecord = templateSel.record.bind(templateSel);
            templateSel.record = function(templateIdx, stateVector) {
                try {
                    // 4軸の平均信用値を計算
                    let sumSuffix = 0, sumEnd = 0, sumPos = 0, sumDelim = 0, count = 0;
                    for (const [, ax] of statTok._axis4) {
                        if (ax.total > 0) {
                            sumSuffix += ax.suffixCount  / ax.total;
                            sumEnd    += ax.wordEndCount / ax.total;
                            sumPos    += ax.posCount > 0 ? Math.min(1, ax.posSum / ax.posCount) : 0;
                            sumDelim  += ax.delimCount   / ax.total;
                            count++;
                        }
                    }
                    // 4軸信用値をstateVectorに追加
                    const axis4Avg = count > 0 ? [
                        sumSuffix / count,
                        sumEnd    / count,
                        sumPos    / count,
                        sumDelim  / count,
                    ] : [0, 0, 0, 0];

                    // 既存stateVector + 4軸平均 = より精密な状態表現
                    const enrichedState = [...(stateVector || []), ...axis4Avg];
                    origRecord(templateIdx, enrichedState);
                } catch(e) {
                    origRecord(templateIdx, stateVector);
                }
            };

            // select() も同じenrichedStateで選ぶようにラップ
            const origSelect = templateSel.select.bind(templateSel);
            templateSel.select = function(stateVector, conceptStr) {
                try {
                    let sumSuffix = 0, sumEnd = 0, sumPos = 0, sumDelim = 0, count = 0;
                    for (const [, ax] of statTok._axis4) {
                        if (ax.total > 0) {
                            sumSuffix += ax.suffixCount  / ax.total;
                            sumEnd    += ax.wordEndCount / ax.total;
                            sumPos    += ax.posCount > 0 ? Math.min(1, ax.posSum / ax.posCount) : 0;
                            sumDelim  += ax.delimCount   / ax.total;
                            count++;
                        }
                    }
                    const axis4Avg = count > 0 ? [
                        sumSuffix / count,
                        sumEnd    / count,
                        sumPos    / count,
                        sumDelim  / count,
                    ] : [0, 0, 0, 0];

                    const enrichedState = [...(stateVector || []), ...axis4Avg];
                    return origSelect(enrichedState, conceptStr);
                } catch(e) {
                    return origSelect(stateVector, conceptStr);
                }
            };

            being.addLog && being.addLog('[PIPE/4軸①] TemplateSelector → 4軸stateVector拡張 完了');
        }

        // ── ② LanguageInputDL の理解深度に4軸を反映 ───────────────────
        // parse()が返すsyntaxInfoに4軸信用値を追加する
        // 「この言語はsuffix型か position型か」を理解に乗せる
        if (langDL && langDL.parse) {
            const origLangParse = langDL.parse.bind(langDL);
            langDL.parse = async function(text, ...args) {
                const result = await origLangParse(text, ...args);
                try {
                    // テキストのトークンから4軸信用値を取得
                    const tokens = statTok._segment ? statTok._segment(text) : [];
                    if (tokens.length > 0) {
                        let sumSuffix = 0, sumEnd = 0, sumPos = 0, sumDelim = 0, count = 0;
                        for (const tok of tokens) {
                            const info = statTok.tokenScores && statTok.tokenScores.get(tok.surface);
                            if (info) {
                                sumSuffix += info.suffixConf   || 0;
                                sumEnd    += info.wordEndConf  || 0;
                                sumPos    += info.positionConf || 0;
                                sumDelim  += info.delimConf    || 0;
                                count++;
                            }
                        }
                        if (count > 0 && result) {
                            result.axis4 = {
                                suffix:   sumSuffix / count,
                                wordEnd:  sumEnd    / count,
                                position: sumPos    / count,
                                delim:    sumDelim  / count,
                            };
                            // 支配的な軸を判定（この言語の文法タイプ）
                            const axes = result.axis4;
                            result.dominantGrammarAxis = Object.entries(axes)
                                .sort((a, b) => b[1] - a[1])[0][0];
                        }
                    }
                } catch(e) { console.warn('[PIPE/4軸②] langDL.parse hook error:', e); }
                return result;
            };

            being.addLog && being.addLog('[PIPE/4軸②] LanguageInputDL → 4軸理解深度 完了');
        }

        // ── ③ episodicMemory.store() に4軸信用値を付与 ─────────────────
        // エピソードに「このとき言語の何軸が強かったか」を記録する
        // 後でretrieve()したとき因果推論野が言語文脈を再現できる
        if (episodic2) {
            const origStore2 = episodic2.store.bind(episodic2);
            episodic2.store = function(fragment) {
                try {
                    // 4軸の現在値をスナップショットとして付与
                    let sumSuffix = 0, sumEnd = 0, sumPos = 0, sumDelim = 0, count = 0;
                    for (const [, ax] of statTok._axis4) {
                        if (ax.total > 0) {
                            sumSuffix += ax.suffixCount  / ax.total;
                            sumEnd    += ax.wordEndCount / ax.total;
                            sumPos    += ax.posCount > 0 ? Math.min(1, ax.posSum / ax.posCount) : 0;
                            sumDelim  += ax.delimCount   / ax.total;
                            count++;
                        }
                    }
                    if (count > 0 && fragment && typeof fragment === 'object') {
                        fragment._axis4Snapshot = {
                            suffix:   sumSuffix / count,
                            wordEnd:  sumEnd    / count,
                            position: sumPos    / count,
                            delim:    sumDelim  / count,
                            timestamp: Date.now(),
                        };
                    }
                } catch(e) {}
                origStore2(fragment);
            };

            being.addLog && being.addLog('[PIPE/4軸③] episodicMemory → 4軸スナップショット付与 完了');
        }

        being.addLog && being.addLog('[PIPE/4軸] 4軸信用値 → 学習系全接続 完了');
    }

    console.log('[PIPE6] クオリア力場→意思決定→CIR→出力生成 パイプ接続完了');
    being.addLog && being.addLog('[PIPE6] 意思→因果推論→出力 パイプ6 接続完了');
}

// ─────────────────────────────────────────────────────────────────────
// 画像生成トリガー（GPU合成 → ImageGenerator）
// ─────────────────────────────────────────────────────────────────────
async function _triggerImageFromDesign(being, design, gpuBlender) {
    try {
        const vectors = design.spatialPairs.map(p => p.vector);
        const weights = design.spatialPairs.map(p => p.weight);

        // GPU合成
        const blended = gpuBlender.blend(vectors, weights);

        // ImageGeneratorのgenerateFromMood()に合成ベクトルを渡す
        const mood = being.state || { joy: 0.5, tension: 0.3, curiosity: 0.5 };

        // 合成ベクトルをspatialHints形式に変換
        const n    = 16;
        const cellW = Math.floor(2104 / (n * n));
        const cellMaps = {
            brightness:     Array.from({ length: n*n }, (_, i) => blended[i * cellW] || 0),
            hue:            Array.from({ length: n*n }, (_, i) => (blended[i * cellW + 1] || 0) * 360),
            hogStr:         Array.from({ length: n*n }, (_, i) => blended[i * cellW + 2] || 0),
            hogDir:         Array.from({ length: n*n }, (_, i) => Math.floor((blended[i * cellW + 3] || 0) * 8)),
            briReliability: Array.from({ length: n*n }, () => 0.7),
            hueReliability: Array.from({ length: n*n }, () => 0.7),
            hogReliability: Array.from({ length: n*n }, () => 0.7),
        };

        const imageHints = [{
            label:              design.text,
            count:              design.spatialPairs.length,
            spatialHints:       { cellMaps },
            meanVector:         Array.from(blended),
            avgBrightness:      blended.slice(0, 256).reduce((a, b) => a + b, 0) / 256,
            brightnessReliability: 0.7,
            hueReliability:     0.7,
            shapeReliability:   0.7,
            textureReliability: 0.6,
            fromGPUBlend:       true,
        }];

        const result = await being.imageGenerator.generateFromMood(mood, imageHints);
        being.addLog && being.addLog(
            `[PIPE6] 画像生成完了: ${design.spatialPairs.map(p=>p.token).join('+')} → GPU合成`
        );
        return result;
    } catch(e) {
        console.warn('[PIPE6] _triggerImageFromDesign error:', e);
        return null;
    }
}

// ─────────────────────────────────────────────────────────────────────
// 音声生成トリガー（因果推論済みのaudioPairs → AudioGenerator）
// ─────────────────────────────────────────────────────────────────────
async function _triggerAudioFromDesign(being, design) {
    try {
        const mood = being.state || { joy: 0.5, tension: 0.3, curiosity: 0.5 };

        // audioPairsを重み付き平均してaudioHints形式に変換
        let totalWeight = 0;
        let avgEntropy = 0, avgPMI = 0, avgTempo = 0, avgBrightness = 0;
        for (const p of design.audioPairs) {
            avgEntropy    += p.avgEntropy    * p.weight;
            avgPMI        += p.avgPMI        * p.weight;
            avgTempo      += p.avgTempo      * p.weight;
            avgBrightness += p.avgBrightness * p.weight;
            totalWeight   += p.weight;
        }
        if (totalWeight > 0) {
            avgEntropy    /= totalWeight;
            avgPMI        /= totalWeight;
            avgTempo      /= totalWeight;
            avgBrightness /= totalWeight;
        }

        const audioHints = [{
            label:         design.text,
            count:         design.audioPairs.length,
            avgEntropy,
            avgPMI,
            avgTempo,
            avgBrightness,
            fromCIRDesign: true,
        }];

        const result = await being.audioGenerator.generateFromMood(mood, audioHints);
        being.addLog && being.addLog(
            `[PIPE6] 音声生成完了: ${design.audioPairs.map(p=>p.token).join('+')} → 因果推論合成`
        );
        return result;
    } catch(e) {
        console.warn('[PIPE6] _triggerAudioFromDesign error:', e);
        return null;
    }
}

// ─────────────────────────────────────────────────────────────────────
// 自動アタッチ（PIPE5完了後）
// ─────────────────────────────────────────────────────────────────────
(function pollForPipe6() {
    const being = window.ao;
    if (being
        && being.qualiaField
        && being.causalInterventionReasoner
        && being.prefrontalCoreV1_1
        && being.imageGenerator
        && being.audioGenerator) {
        setTimeout(() => {
            try { attachPipe6(being); } catch(e) { console.error('[PIPE6] error:', e); }
        }, 3500);
    } else {
        setTimeout(pollForPipe6, 1000);
    }
})();

// ─────────────────────────────────────────────────────────────────────
// 動画生成トリガー
// CIRが設計したシーン列 → GPU並列フレーム生成 → シーン統合 → 動画
// ─────────────────────────────────────────────────────────────────────
async function _triggerVideoFromDesign(being, design, gpuBlender) {
    try {
        const { scenes, duration, fps } = design;
        const totalFrames = duration * fps;
        const W = 1280, H = 720;

        const canvas = document.createElement('canvas');
        canvas.width  = W;
        canvas.height = H;
        const ctx = canvas.getContext('2d');
        const allFrames = [];

        being.addLog && being.addLog(
            `[PIPE6/Video] シーン並列生成開始: ${scenes.length}シーン × ${fps}fps`
        );

        // ── GPU並列：各シーンのspatialベクトルを事前合成 ────────────
        // シーンごとにGPU合成したspatialを持っておく
        const sceneBlended = await Promise.all(scenes.map(async (scene, si) => {
            const vectors = [];
            const weights = [];

            // このシーンのspatialベクトル
            if (scene.spatial) {
                vectors.push(scene.spatial);
                weights.push(scene.confidence);
            }

            // ConceptGraphから親カテゴリのspatialも混ぜる
            if (being.conceptGraph) {
                const parents = being.conceptGraph.getParents(scene.label);
                for (const parent of parents) {
                    const parentHyp = being.imageAdapter &&
                        being.imageAdapter.hypothesisTable &&
                        being.imageAdapter.hypothesisTable.hypotheses &&
                        being.imageAdapter.hypothesisTable.hypotheses.get(parent);
                    if (parentHyp && parentHyp.spatial && parentHyp.spatial.samples >= 4) {
                        const n = parentHyp.spatial.samples;
                        vectors.push(new Float32Array(parentHyp.spatial.sum.map(v => v / n)));
                        weights.push((parentHyp.confidence || 0.3) * 0.5);
                    }
                }
            }

            // GPU合成
            const blended = vectors.length > 0
                ? gpuBlender.blend(vectors, weights)
                : null;

            return { ...scene, blended, sceneIndex: si };
        }));

        being.addLog && being.addLog('[PIPE6/Video] GPU並列合成完了 → フレーム描画開始');

        // ── シーンごとにフレームを生成して統合 ───────────────────────
        for (let si = 0; si < sceneBlended.length; si++) {
            const scene     = sceneBlended[si];
            const nextScene = sceneBlended[si + 1] || null;
            const frameCount = scene.frameCount;

            for (let fi = 0; fi < frameCount; fi++) {
                const progress    = fi / frameCount;         // シーン内進捗 0→1
                const globalProg  = (si * frameCount + fi) / totalFrames;
                const wave        = Math.sin(globalProg * Math.PI * 2);

                // シーン末尾でブレンド遷移（因果推論が決めたtransition）
                const blendRatio = (scene.transition === 'blend' && fi < 6)
                    ? fi / 6   // 最初の6フレームで前シーンからフェード
                    : (nextScene && fi >= frameCount - 6)
                        ? (frameCount - fi) / 6  // 最後の6フレームで次シーンへフェード
                        : 1.0;

                // spatialベクトルをcanvasに描画
                if (scene.blended) {
                    _drawSpatialToCanvas(ctx, scene.blended, W, H, wave, blendRatio);
                } else {
                    // spatialなし → 気分ベースのフォールバック
                    const hue = (si / scenes.length) * 360;
                    ctx.fillStyle = `hsl(${hue}, 60%, ${20 + blendRatio * 50}%)`;
                    ctx.fillRect(0, 0, W, H);
                }

                // シーンラベルをオーバーレイ（薄く）
                ctx.fillStyle = `rgba(255,255,255,${0.3 * blendRatio})`;
                ctx.font = '36px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(scene.label, W / 2, H - 60);

                allFrames.push(canvas.toDataURL('image/jpeg', 0.8));

                if (window.taskManager) {
                    window.taskManager.updateProgress('generate',
                        (si * frameCount + fi + 1) / totalFrames);
                }
            }
        }

        being.addLog && being.addLog(
            `[PIPE6/Video] 動画生成完了: ${allFrames.length}フレーム`
        );

        return {
            frames:      allFrames,
            fps,
            duration,
            description: `${design.text} (因果推論設計 ${scenes.length}シーン)`,
            scenes:      sceneBlended.map(s => s.label),
        };

    } catch(e) {
        console.warn('[PIPE6/Video] _triggerVideoFromDesign error:', e);
        return null;
    }
}

// spatialベクトルをcanvasに描画（ImageGeneratorと同じ方式）
function _drawSpatialToCanvas(ctx, spatial, W, H, wave, alpha) {
    const GCELLS = 16;
    const cw = W / GCELLS, ch = H / GCELLS;
    const hueHist        = spatial.slice(0, 8);
    const brightnessGrid = spatial.slice(8, 264);
    const hogBlocks      = spatial.slice(272, 2320);
    const dirs = [0, Math.PI/8, Math.PI/4, 3*Math.PI/8,
                  Math.PI/2, 5*Math.PI/8, 3*Math.PI/4, 7*Math.PI/8];

    const domHueIdx = hueHist.indexOf(Math.max(...hueHist));
    const domHue    = (domHueIdx / 8) * 360;
    const bAvg      = brightnessGrid.reduce((a, b) => a + b, 0) / brightnessGrid.length;

    ctx.globalAlpha = alpha;

    for (let gy = 0; gy < GCELLS; gy++) {
        for (let gx = 0; gx < GCELLS; gx++) {
            const ci        = gy * GCELLS + gx;
            const b         = brightnessGrid[ci] || 0;
            const blockBase = ci * 8;
            const hogSlice  = hogBlocks.slice(blockBase, blockBase + 8);
            const maxStr    = Math.max(...hogSlice);
            const domDir    = hogSlice.indexOf(maxStr);

            const lit = Math.min(90, Math.max(10,
                bAvg * 100 + (b - bAvg) * 80 + wave * 3
            ));
            ctx.fillStyle = `hsl(${domHue}, 50%, ${lit}%)`;
            ctx.fillRect(gx * cw, gy * ch, cw + 1, ch + 1);

            if (maxStr > 0.1) {
                const angle = dirs[domDir] + wave * 0.05;
                const cx2   = gx * cw + cw / 2;
                const cy2   = gy * ch + ch / 2;
                const len   = cw * 0.8 * maxStr;
                ctx.strokeStyle = `hsla(${domHue}, 50%, ${lit > 50 ? 20 : 80}%, ${maxStr})`;
                ctx.lineWidth   = 1 + maxStr * 2;
                ctx.beginPath();
                ctx.moveTo(cx2 - Math.cos(angle)*len, cy2 - Math.sin(angle)*len);
                ctx.lineTo(cx2 + Math.cos(angle)*len, cy2 + Math.sin(angle)*len);
                ctx.stroke();
            }
        }
    }
    ctx.globalAlpha = 1.0;
}



// ─────────────────────────────────────────────────────────────────────
// CIR設計 → 各編集タイプのアクションに変換
// ハードコードなし：CIRのトークン・spatialPairs・audioPairsを使う
// ─────────────────────────────────────────────────────────────────────

// コード編集：CIRトークンをコード構造に変換
function _cirToCodeActions(design, task, assets) {
    const actions = [];
    for (const tok of design.tokens) {
        // spatialPairsがあれば構造的なコードブロックとして挿入
        const hasSpatial = design.spatialPairs.find(p => p.token === tok);
        if (hasSpatial) {
            actions.push({
                type:        'insert',
                description: `${tok}に関するコードブロック`,
                confidence:  hasSpatial.weight,
                fromCIR:     true,
            });
        } else {
            actions.push({
                type:        'comment',
                description: `// ${tok}`,
                fromCIR:     true,
            });
        }
    }
    if (actions.length === 0) actions.push({ type: 'open', file: assets[0] || 'main.js' });
    return actions;
}

// 文章編集：CIRトークン + ConceptGraphで段落構造を設計
function _cirToDocActions(design, task, assets) {
    const actions = [];
    const cg = window._aoConceptGraph;

    for (const tok of design.tokens) {
        // ConceptGraphに概念があれば見出しとして使う
        const isCategory = cg && cg.groups.has(tok);
        if (isCategory) {
            actions.push({
                type:        'format_heading',
                level:        1,
                text:         tok,
                members:      [...(cg.groups.get(tok) || [])],
                fromCIR:      true,
            });
            // カテゴリメンバーを箇条書きとして追加
            const members = [...(cg.groups.get(tok) || [])];
            for (const m of members) {
                actions.push({
                    type:    'insert_paragraph',
                    text:    m,
                    fromCIR: true,
                });
            }
        } else {
            actions.push({
                type:    'insert_paragraph',
                text:    tok,
                fromCIR: true,
            });
        }
    }
    if (actions.length === 0) actions.push({ type: 'create', template: 'blank' });
    return actions;
}

// 音声編集：CIRのaudioPairs → AudioGeneratorの出力をアクションに
function _cirToAudioActions(design, task, assets, being) {
    const actions = [];

    // audioPairsがあれば音響パラメータをアクションに変換
    for (const ap of design.audioPairs) {
        actions.push({
            type:         'apply_effect',
            effect:       ap.avgEntropy > 0.5 ? 'reverb' : 'eq',
            token:        ap.token,
            avgEntropy:   ap.avgEntropy,
            avgPMI:       ap.avgPMI,
            avgTempo:     ap.avgTempo,
            avgBrightness: ap.avgBrightness,
            fromCIR:      true,
        });
    }

    // AudioGeneratorで生成してから編集
    if (being && being.audioGenerator && design.audioPairs.length > 0) {
        actions.push({
            type:        'generate_and_import',
            description: `${task}の音声をAoが生成してインポート`,
            fromCIR:     true,
            async:       true,
        });
    }

    if (actions.length === 0) actions.push({ type: 'import', file: assets[0] || 'audio.wav' });
    return actions;
}

// スライド編集：12軸の状態 → スライド構成に変換
function _cirToSlideActions(design, task, assets, being) {
    const actions = [];
    const wv = being && being.worldView;

    // 12軸の重要度でスライド枚数・構成を決める
    const hierarchyScore = wv ? (wv.getAxis('hierarchy') || 0) : 0.5;
    const slideCount = Math.max(3, Math.ceil(design.tokens.length * (1 + hierarchyScore)));

    // タイトルスライド
    actions.push({
        type:    'add_slide',
        layout:  'title',
        text:    task,
        fromCIR: true,
    });

    // CIRトークンごとにコンテンツスライド
    for (const tok of design.tokens) {
        const cg = window._aoConceptGraph;
        const members = cg ? [...(cg.groups.get(tok) || [])] : [];

        actions.push({
            type:        'add_slide',
            layout:      members.length > 0 ? 'bullet' : 'content',
            text:        tok,
            bullets:     members,
            axisScore:   hierarchyScore,
            fromCIR:     true,
        });
    }

    // エンディングスライド
    actions.push({
        type:    'add_slide',
        layout:  'blank',
        fromCIR: true,
    });

    // テーマを12軸から決める
    const causalityScore = wv ? (wv.getAxis('causality') || 0) : 0.5;
    actions.push({
        type:    'apply_theme',
        theme:   causalityScore > 0.6 ? 'professional' : 'minimal',
        fromCIR: true,
    });

    return actions;
}

// 動画編集：CIR設計をVideoActionsに変換
function _cirToVideoActions(design, task, assets) {
    const actions = [];
    for (const sp of design.spatialPairs) {
        actions.push({
            type:        'import',
            token:       sp.token,
            confidence:  sp.weight,
            fromCIR:     true,
        });
    }
    actions.push({ type: 'arrange', layout: 'timeline', fromCIR: true });
    if (actions.length === 1) actions.unshift({ type: 'import', assets, fromCIR: true });
    return actions;
}

window.GPUSpatialBlender = GPUSpatialBlender;
window.attachPipe6       = attachPipe6;
