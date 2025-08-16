-- Small additional table for RAG testing: Patient Medications
CREATE OR REPLACE TABLE HEALTHDB.PATIENT_MEDICATIONS (
    MEDICATION_ID VARCHAR(100),
    MEMBER_ID VARCHAR(200),
    DRUG_NAME VARCHAR(300),
    DOSAGE VARCHAR(100),
    FREQUENCY VARCHAR(100),
    START_DATE DATE,
    END_DATE DATE,
    PRESCRIBING_DOCTOR VARCHAR(300),
    STATUS VARCHAR(50),
    NOTES VARCHAR(500)
);

-- Insert sample medication data
INSERT INTO HEALTHDB.PATIENT_MEDICATIONS VALUES
('MED001', 'MBR12345', 'Tamoxifen', '20mg', 'Once daily', '2024-12-01', '2025-12-01', 'Dr. Sarah Johnson', 'Active', 'For breast cancer treatment'),
('MED002', 'MBR12345', 'Ondansetron', '8mg', 'As needed', '2024-12-01', '2025-03-01', 'Dr. Sarah Johnson', 'Active', 'Anti-nausea medication'),
('MED003', 'MBR67890', 'Carboplatin', '300mg/m2', 'Every 3 weeks', '2024-11-15', '2025-02-15', 'Dr. Emily Rodriguez', 'Active', 'Chemotherapy agent'),
('MED004', 'MBR24681', 'Prednisone', '10mg', 'Twice daily', '2024-10-20', '2024-11-20', 'Dr. Emily Rodriguez', 'Completed', 'Steroid for inflammation'),
('MED005', 'MBR13579', 'Metformin', '500mg', 'Twice daily', '2023-06-01', NULL, 'Dr. Amanda Foster', 'Active', 'Diabetes management'),
('MED006', 'MBR98765', 'Atorvastatin', '40mg', 'Once daily at bedtime', '2023-01-15', NULL, 'Dr. Thomas Wilson', 'Active', 'Cholesterol management'),
('MED007', 'MBR67890', 'Lorazepam', '0.5mg', 'As needed', '2024-11-15', '2025-02-15', 'Dr. Emily Rodriguez', 'Active', 'Anxiety management during treatment');

-- Small lookup table for drug categories
CREATE OR REPLACE TABLE HEALTHDB.DRUG_CATEGORIES (
    DRUG_NAME VARCHAR(300),
    CATEGORY VARCHAR(200),
    DRUG_CLASS VARCHAR(200),
    REQUIRES_MONITORING BOOLEAN
);

INSERT INTO HEALTHDB.DRUG_CATEGORIES VALUES
('Tamoxifen', 'Oncology', 'Hormone Therapy', TRUE),
('Ondansetron', 'Supportive Care', 'Antiemetic', FALSE),
('Carboplatin', 'Oncology', 'Chemotherapy', TRUE),
('Prednisone', 'Supportive Care', 'Corticosteroid', TRUE),
('Metformin', 'Endocrine', 'Antidiabetic', FALSE),
('Atorvastatin', 'Cardiovascular', 'Statin', FALSE),
('Lorazepam', 'Supportive Care', 'Benzodiazepine', TRUE);