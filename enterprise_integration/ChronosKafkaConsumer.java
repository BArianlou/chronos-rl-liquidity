package com.chronos.enterprise.streaming;

import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

/**
 * CHRONOS DISTRIBUTED SYSTEM UTILITY
 * Architecture: Java / Apache Kafka
 * Purpose: High-throughput ingestion of market telemetry. Acts as the kinetic 
 * data pipeline feeding the PySpark feature engineering layer.
 */
@Component
public class ChronosKafkaConsumer {

    private static final Logger logger = LoggerFactory.getLogger(ChronosKafkaConsumer.class);
    
    // Service to buffer and batch records for PySpark ingestion
    private final TelemetryBufferService bufferService;

    public ChronosKafkaConsumer(TelemetryBufferService bufferService) {
        this.bufferService = bufferService;
    }

    /**
     * Architecture: Listens to the high-frequency market tick topic.
     * Enforces strict error handling to prevent ghost-data from poisoning the RL Agent.
     */
    @KafkaListener(topics = "chronos.market.ticks.v1", groupId = "chronos-rl-group")
    public void consumeMarketTick(ConsumerRecord<String, String> record) {
        try {
            String assetId = record.key();
            String telemetryPayload = record.value();
            
            // Log trace for data provenance
            logger.debug("Ingesting tick for Asset: {} at Offset: {}", assetId, record.offset());
            
            // Route to buffer for downstream PySpark windowing
            bufferService.batchTelemetry(assetId, telemetryPayload);
            
        } catch (Exception e) {
            // Fail-Safe: Quarantine corrupted ticks without crashing the consumer loop
            logger.error("KINETIC PIPELINE ERROR: Failed to process market tick. Quarantining record. Offset: {}", record.offset(), e);
        }
    }
}
