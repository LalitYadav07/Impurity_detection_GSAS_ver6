(function (root, factory) {
  const plugin = factory();
  if (typeof module === "object" && module.exports) module.exports = plugin;
  else root.radarTransitionCursor = plugin;
})(typeof window === "undefined" ? globalThis : window, function () {
  "use strict";

  // Local input owns the thumb until its matching acknowledgement arrives.
  // Neither an older server selection nor an older gesture may move it.
  class CursorState {
    constructor(clientId) {
      this.clientId = clientId;
      this.revision = 0;
      this.submitted = 0;
      this.index = 0;
      this.dragging = false;
      this.pending = false;
      this.bundle = null;
      this.target = null;
    }
    owns(scope) {
      return Boolean(this.bundle && this.bundle.scope === scope && (this.dragging || this.pending));
    }
    begin(server, pointer) {
      if (!this.owns(server.scope)) {
        this.bundle = server;
        this.index = server.cursor;
      }
      this.revision += 1;
      this.dragging = pointer;
      this.pending = true;
      this.target = this.bundle.runIds[this.index];
      return {client_id: this.clientId, revision: this.revision, scope: server.scope};
    }
    input(index) {
      if (!this.bundle || !Number.isInteger(index) || index < 0 || index >= this.bundle.runIds.length) return false;
      this.index = index;
      this.target = this.bundle.runIds[index];
      return true;
    }
    release() {
      this.dragging = false;
      if (!this.bundle || !this.target || this.submitted === this.revision) return null;
      this.submitted = this.revision;
      return {client_id: this.clientId, revision: this.revision, scope: this.bundle.scope,
        epoch: this.bundle.epoch, index: this.index, run_id: this.target};
    }
    reconcile(server, ack) {
      if (this.bundle && this.bundle.scope !== server.scope) {
        this.dragging = false;
        this.pending = false;
        this.bundle = null;
      }
      if (this.dragging) return false;
      if (this.pending) {
        if (!ack || ack.client_id !== this.clientId || ack.revision !== this.revision) return false;
        if (ack.accepted && server.runIds[server.cursor] !== this.target) return false;
        this.pending = false;
        this.bundle = null;
      }
      this.index = server.cursor;
      return true;
    }
  }

  // Plotly.react is asynchronous. Keep one draw in flight and only the newest
  // waiting frame, rather than replaying a backlog after the pointer reverses.
  class LatestRenderer {
    constructor(draw, complete, failure) {
      this.draw = draw;
      this.complete = complete;
      this.failure = failure;
      this.next = null;
      this.running = false;
      this.stopped = false;
      this.done = Promise.resolve();
    }
    enqueue(packet) {
      if (this.stopped) return this.done;
      this.next = packet;
      if (!this.running) this.done = this.drain();
      return this.done;
    }
    async drain() {
      this.running = true;
      try {
        while (this.next && !this.stopped) {
          const packet = this.next;
          this.next = null;
          try {
            await this.draw(packet);
            if (!this.stopped) this.complete(packet);
          } catch (error) {
            if (!this.stopped) this.failure(error);
          }
        }
      } finally {
        this.running = false;
      }
    }
    stop() { this.stopped = true; this.next = null; }
  }

  const clone = (value) => JSON.parse(JSON.stringify(value));
  const component = {
    props: {
      figure: {type: Object, required: true}, frames: {type: Array, default: () => []},
      runIds: {type: Array, default: () => []}, cursor: {type: Number, default: 0},
      scope: {type: String, default: ""}, epoch: {type: String, default: ""},
      ack: {type: Object, default: () => ({})}, top: {type: Number, default: 140},
      height: {type: Number, default: 190},
    },
    emits: ["Begin", "Commit", "Click", "Relayout", "Reset", "Preview"],
    data() {
      const id = window.crypto && window.crypto.randomUUID ? window.crypto.randomUUID() : String(Date.now()) + Math.random();
      return {model: new CursorState(id), failure: "", missingPreview: false};
    },
    computed: {
      activeIds() { return this.model.owns(this.scope) ? this.model.bundle.runIds : this.runIds; },
      viewTop() { return this.model.owns(this.scope) ? this.model.bundle.top : this.top; },
      viewHeight() { return this.model.owns(this.scope) ? this.model.bundle.height : this.height; },
      viewFigureHeight() { return ((this.model.owns(this.scope) ? this.model.bundle.figure : this.figure).layout.height || 485); },
      cursorLabel() {
        const b = this.model.owns(this.scope) ? this.model.bundle : this.snapshot();
        const frame = b.frames[this.model.index];
        return frame ? frame.label : b.runIds[this.model.index] || "Waiting for scans";
      },
    },
    watch: {
      figure() { this.serverChanged(); }, frames() { this.serverChanged(); },
      cursor() { this.serverChanged(); }, scope() { this.serverChanged(); },
      ack() { this.serverChanged(); }, runIds() { this.serverChanged(); },
    },
    mounted() {
      this._listening = false;
      this._renderKey = "";
      this._dimensions = "";
      this._renderer = new LatestRenderer(
        async (packet) => {
          const plot = this.$refs.plot;
          const trace = packet.figure.data[1];
          if (this._baseKey === packet.baseKey && trace && packet.figure.layout.yaxis2) {
            // Cursor-only changes retain the heatmap raster and live x zoom.
            // Update its ribbon trace and selection marks, not every map cell.
            const layout = packet.figure.layout;
            await window.Plotly.update(plot,
              {x:[clone(trace.x)],y:[clone(trace.y)],customdata:[clone(trace.customdata)],
                name:trace.name,hovertemplate:trace.hovertemplate},
              clone({annotations:layout.annotations || [],shapes:layout.shapes || [],
                "yaxis2.tickvals":layout.yaxis2.tickvals,"yaxis2.ticktext":layout.yaxis2.ticktext}),[1]);
            this.$refs.input.dataset.plotUpdateKind = "selection";
          } else {
            await window.Plotly.react(plot, clone(packet.figure.data), clone(packet.figure.layout),
              {responsive: false, displayModeBar: "hover", displaylogo: false, doubleClick: false});
            this._baseKey = packet.baseKey;
            this.$refs.input.dataset.plotUpdateKind = "map";
          }
          if (!this._listening) {
            plot.on("plotly_click", (event) => {
              const point = event && event.points && event.points[0];
              if (point && point.data.type === "heatmap" && !this.model.dragging) this.$emit("Click", point.y);
            });
            plot.on("plotly_relayout", (event) => this.$emit("Relayout", event));
            plot.on("plotly_doubleclick", () => this.$emit("Reset"));
            this._listening = true;
          }
          const dimensions = plot.clientWidth + ":" + plot.clientHeight;
          if (dimensions !== this._dimensions) {
            this._dimensions = dimensions;
            await window.Plotly.Plots.resize(plot);
          }
        },
        (packet) => {
          const input = this.$refs.input;
          input.dataset.renderedRunId = packet.runId || "";
          input.dataset.renderedRevision = String(packet.revision);
          if (packet.started) input.dataset.previewLatencyMs = (window.performance.now() - packet.started).toFixed(1);
        },
        (error) => { this.failure = "The pattern preview could not be drawn. Select the scan again."; console.error(error); });
      this._observer = new ResizeObserver(() => this.enqueue(true));
      this._observer.observe(this.$refs.plot);
      this.serverChanged();
    },
    beforeUnmount() {
      const plot = this.$refs.plot;
      this._renderer.stop();
      this._observer.disconnect();
      this._renderer.done.finally(() => window.Plotly.purge(plot));
    },
    methods: {
      snapshot() {
        return {scope: this.scope, epoch: this.epoch, figure: this.figure,
          frames: this.frames, runIds: this.runIds, cursor: this.cursor, top: this.top, height: this.height};
      },
      serverChanged() {
        if (this._serverSyncQueued) return;
        this._serverSyncQueued = true;
        this.$nextTick(() => {
          this._serverSyncQueued = false;
          if (this.model.reconcile(this.snapshot(), this.ack)) {
            this.$emit("Preview", {active: false});
            this.enqueue();
          } else {
            this.trace("hold");
          }
        });
      },
      begin(event) {
        this._started = window.performance.now();
        const message = this.model.begin(this.snapshot(), true);
        this.model.input(Number(event.target.value));
        event.target.setPointerCapture(event.pointerId);
        this.$emit("Begin", message);
      },
      input(event) {
        this._started = window.performance.now();
        if (!this.model.dragging) this.$emit("Begin", this.model.begin(this.snapshot(), false));
        if (this.model.input(Number(event.target.value))) {
          this.trace("input");
          const frame = this.model.bundle.frames[this.model.index];
          this.$emit("Preview", frame ? {active: true, label: frame.label, notice: frame.notice, evidence: frame.evidence} : {active: false});
          this.enqueue();
        }
      },
      change() { if (!this.model.dragging) this.release(); },
      release() {
        const message = this.model.release();
        if (message) { this.trace("release"); this.$emit("Commit", message); }
      },
      trace(action) {
        const input = this.$refs.input;
        if (!input) return;
        this._trace = [...(this._trace || []), {action, index: this.model.index, value: Number(input.value),
          revision: this.model.revision, serverCursor: this.cursor,
          ackRevision: this.ack.revision == null ? null : this.ack.revision}].slice(-64);
        input.dataset.cursorTrace = JSON.stringify(this._trace);
      },
      enqueue(force) {
        if (!this._renderer || !this.$refs.plot) return;
        const owns = this.model.owns(this.scope);
        const bundle = owns ? this.model.bundle : this.snapshot();
        const index = owns ? this.model.index : this.cursor;
        const frame = owns && bundle.frames[index];
        this.missingPreview = Boolean(owns && !frame);
        if (this.missingPreview) return;
        // An acknowledgement of the already previewed run needs no second
        // drawing. Epoch changes cover data, display settings and zoom resets.
        const key = bundle.scope + ":" + bundle.epoch + ":" + bundle.runIds[index];
        if (!force && key === this._renderKey) return;
        this._renderKey = key;
        let figure = bundle.figure;
        if (frame) figure = {data: [figure.data[0], frame.trace], layout: {...figure.layout,
          annotations: frame.annotations, shapes: frame.shapes,
          yaxis2: {...figure.layout.yaxis2, ...frame.axis}}};
        this._renderer.enqueue({figure, runId: bundle.runIds[index], revision: this.model.revision,
          baseKey:bundle.scope+":"+bundle.epoch,started: frame ? this._started : null});
      },
    },
    template: `<div>
      <div class="radar-transition-cursor-map">
        <div class="radar-transition-scan-cursor">
          <label for="transition-scan-cursor" style="position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%)">Scan cursor</label>
          <input ref="input" id="transition-scan-cursor" type="range" min="0" :max="Math.max(0,activeIds.length-1)" step="1"
            :value="model.index" :aria-valuetext="cursorLabel" :disabled="activeIds.length<2"
            :style="{position:'absolute',writingMode:'vertical-lr',direction:'ltr',accentColor:'#15543c',width:'24px',margin:0,top:viewTop+'px',height:viewHeight+'px'}"
            @pointerdown="begin" @input="input" @change="change" @pointerup="release" @pointercancel="release" @lostpointercapture="release" @blur="release">
        </div>
        <div class="radar-transition-map-frame" :style="{height:viewFigureHeight+'px'}">
          <div ref="plot" style="width:100%;height:100%"></div>
        </div>
      </div>
      <p v-if="missingPreview" class="radar-section-help">Release the cursor to load this scan.</p>
      <p v-if="failure" role="alert" class="radar-section-help">{{failure}}</p>
    </div>`,
  };
  return {CursorState, LatestRenderer, install(app) { app.component("radar-transition-cursor", component); }};
});
