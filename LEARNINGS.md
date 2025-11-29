# LEARNINGS.md

## README Design Philosophy (2024-11)

### Production-First Documentation
- README should represent the **ideal state** of the module, not current limitations
- Focus on **practical use cases** across deployment scenarios
- Document architecture patterns, not just API
- Include performance characteristics and trade-offs

### Key Use Cases Identified
1. **Edge/IoT**: MFCC-only, ultra-low latency, privacy-first
2. **Development**: Bundled Whisper, zero-config quick start
3. **Production LAN**: Dedicated Whisper server, scalable, low-latency
4. **Cloud**: Managed infrastructure, highest scale
5. **Smart Home**: Multi-detector, async callbacks, event-driven
6. **Voice Assistant**: Full pipeline with command processing

### Architecture Insights
- **Three-tier detection**: Pre-filter (MFCC) → Confirmation (Whisper) → Action
- MFCC runs < 10ms, eliminates ~95% of non-matches before expensive transcription
- Latency budget: MFCC < 10ms, LAN Whisper ~200ms, Cloud ~1-2s
- False positive reduction: MFCC alone ~5-10%, with Whisper < 1%

### Deployment Patterns Documented
1. **Edge with fallback**: Try LAN, fall back to MFCC-only
2. **Load balancing**: Multiple Whisper servers, round-robin
3. **Multi-stage**: Fast MFCC filter → Whisper confirmation

### Documentation Best Practices
- Lead with **deployment architecture** before code examples
- Provide **6+ real-world examples** covering major use cases
- Include **performance tables** (latency, accuracy, resource usage)
- Add **comprehensive FAQ** addressing deployment, technical, and troubleshooting
- Show **full production patterns**, not just toy examples
- Include **troubleshooting guide** with actionable diagnostics

### Technical Documentation
- Document the **detection pipeline** with state machine
- Include **architecture diagram** (ASCII art for universality)
- Explain **MFCC vs Whisper trade-offs** clearly
- Provide **tuning guidance** for similarity_threshold
- Show how to **create reference audio** properly

## Module Design Goals

### STT Backend Flexibility
- `stt_backend=None`: MFCC-only (no network, ultra-fast)
- `stt_backend="bundled"`: Auto-download mini_transcriber (zero-config)
- `external_whisper_url`: Point to LAN/cloud server (production)

### API Design
- **Sync API**: `waitforit()` for simple blocking detection
- **Async API**: `start()`/`stop()` with callbacks for event-driven
- **Resource cleanup**: Proper cleanup in `stop()` and `__del__()`

### Performance Targets
- MFCC matching: < 10ms
- Memory footprint: ~50MB (detection only)
- CPU usage: 2-5% steady-state
- Buffer: 10 second rolling window (configurable)

## Future Enhancements (From README Review)

### Code Improvements Needed
1. Make buffer size configurable (currently hardcoded 10s) [not implemented]
2. Add session configuration for Whisper API headers (authentication) [not implemented]
3. Add verbose logging option (currently hardcoded print statements) [not implemented]
4. Expose MFCC parameters (n_mfcc, n_fft, hop_length) as constructor args [not implemented]
5. Add metrics/telemetry (detection rate, latency, false positives) [not implemented]
6. Add health check method for transcription service [not implemented]
7. Add retry logic for transient network failures [not implemented]

### Documentation Improvements
1. Add video tutorial for creating reference audio [not implemented]
2. Add benchmarks section with real hardware results [not implemented]
3. Add security considerations section [not implemented]
4. Add deployment checklist (firewall, ports, etc.) [not implemented]

### Testing Improvements
1. Add integration tests with actual audio [not implemented]
2. Add performance benchmarks [not implemented]
3. Add multi-platform CI (Windows, Linux, macOS) [not implemented]
4. Add stress tests (continuous operation, memory leaks) [not implemented]

## Development Workflow

- Always update LEARNINGS.md when solving new problems
- Check LEARNINGS.md before implementing to avoid reinventing
- Document architecture decisions with rationale
- Include performance implications in decisions
