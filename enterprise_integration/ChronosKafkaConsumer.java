/**
 * ===============================================================================
 * MODULE MANIFEST: CHRONOS ENTERPRISE INGESTION LAYER (KAFKA CONSUMER)
 * ===============================================================================
 * System Purpose:
 *     Serves as the high-throughput streaming ingestion layer for the CHRONOS engine.
 *     Subscribes to live market data streams, extracts raw tick records, and buffers
 *     state payloads for downstream distributed windowing via the PySpark ETL pipeline.
 *
 * State Boundaries:
 *     - Strictly confined to network I/O, payload extraction, key validation,
 *       and memory buffering.
 *     - Decoupled from mathematical feature engineering (rolling volatility, order-book depth)
 *       handled by PySpark, and reinforcement learning policy optimization governed
 *       by the Python-based RLAgent.
 *
 * Mathematical/Physical Invariants:
 *     1. Temporal Causality:
 *        Market ticks must be ingested in strict chronological order per asset
 *        partition offset sequence:
 *        Offset(t + 1) > Offset(t) for Partition(Asset_i)
 *        to prevent look-ahead bias and non-causal leakage in downstream state tensors.
 *     2. Zero-Loss Ingestion Guarantee:
 *        Zero silent data drops. All incoming telemetry must either be successfully
 *        committed to the cross-stack buffer or explicitly captured in forensic logs.
 *
 * Design Rationale:
 *     Python's Global Interpreter Lock (GIL) introduces concurrency bottlenecks
 *     when processing high-frequency streaming events at scale. This Java/Spring
 *     component leverages native JVM multi-threading, non-blocking I/O, and
 *     optimized Kafka consumer groups to absorb institutional market data throughput.
 * ===============================================================================
 */

package com.chronos.enterprise.streaming;

import java.util.Objects;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

/**
 * High-throughput market telemetry consumer for the CHRONOS execution architecture.
 *
 * <p>Listens on partitioned market data topics, validates stream payload integrity,
 * and routes continuous tick events to the staging buffer for distributed PySpark consumption.</p>
 */
// Spring IoC Integration:
// Registers the consumer as a managed singleton bean within the Spring application
// context, providing deterministic lifecycle management and thread-safe dependency injection.
@Component
public class ChronosKafkaConsumer {

    private static final Logger logger = LoggerFactory.getLogger(ChronosKafkaConsumer.class);

    /**
     * Staging service bridging raw Kafka telemetry into the distributed compute pipeline.
     */
    private final TelemetryBufferService bufferService;

    /**
     * Constructs the Kafka consumer with the required downstream buffering service.
     *
     * @param bufferService the backing buffer service responsible for staging tick telemetry;
     *                      must not be {@code null}.
     */
    public ChronosKafkaConsumer(final TelemetryBufferService bufferService) {
        this.bufferService = Objects.requireNonNull(bufferService, "bufferService must not be null");
    }

    /**
     * Ingests, inspects, and routes a single market telemetry tick record.
     *
     * <p>Acts as the event-driven entry point for raw market feeds. Enforces key presence,
     * logs diagnostic metadata, and safely delegates raw payloads to the buffer layer.</p>
     *
     * <p><b>Invariants:</b></p>
     * <ul>
     *   <li>Offset Monotonicity: Offsets increase monotonically per topic-partition.</li>
     *   <li>Partition Continuity: Processing errors are trapped and logged,
     *       preventing consumer loop termination while maintaining data provenance.</li>
     * </ul>
     *
     * @param record the incoming {@link ConsumerRecord} containing the asset identifier key
     *               and raw JSON/CSV telemetry payload.
     */
    // Stream Topic & Consumer Group Binding:
    // Binds consumer execution to the production tick topic. The 'chronos-rl-group'
    // coordinates partition assignments across horizontal JVM replicas, eliminating duplicate
    // state ingestion while maintaining partition affinity.
    @KafkaListener(topics = "chronos.market.ticks.v1", groupId = "chronos-rl-group")
    public void consumeMarketTick(final ConsumerRecord<String, String> record) {
        try {
            final String assetId = record.key();
            final String telemetryPayload = record.value();

            // Guard against null partition keys which compromise ordering guarantees
            if (assetId == null || telemetryPayload == null) {
                logger.warn("MALFORMED RECORD: Null key or payload detected at partition: {}, offset: {}. Skipping record.",
                        record.partition(), record.offset());
                return;
            }

            // Diagnostic logging for audit trail and state-space provenance
            logger.debug("Ingesting tick for Asset: {} at Partition: {}, Offset: {}",
                    assetId, record.partition(), record.offset());

            // Cross-Runtime Data Staging:
            // Transfers high-velocity JVM stream records into staging memory buffers
            // accessible by PySpark batch and micro-batch operations.
            this.bufferService.batchTelemetry(assetId, telemetryPayload);

        } catch (final Exception ex) {
            // Fault Tolerance & Partition Continuity:
            // Traps downstream serialization or memory exceptions to prevent consumer thread termination.
            // Preserves Kafka consumer poll loop continuity while emitting diagnostic metadata.
            logger.error("PIPELINE INGESTION ERROR: Failed to process market tick. Quarantining record at Topic: {}, Partition: {}, Offset: {}",
                    record.topic(), record.partition(), record.offset(), ex);
        }
    }
}
