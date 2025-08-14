--liquibase formatted sql
--changeset RAML:1ab111ad-57fe-472c-b4ed-ad5c184e70e5  runOnChange:true



CREATE OR REPLACE TABLE HEALTHDB.MEMBER_APPOINTMENTS (
    SCHEDULE_ID VARCHAR(200),
    MEMBER_ID VARCHAR(200),
    APPOINTMENT_SESSION_ID VARCHAR(200),
    GROUP_ID VARCHAR(100),
    SEQUENCE_NUM INTEGER,
    IS_LATEST_SESSION BOOLEAN,
    ROLE_TITLE VARCHAR(400),
    START_DATETIME TIMESTAMP,
    END_DATETIME TIMESTAMP,
    DURATION_MINUTES INTEGER,
    STATUS_TITLE VARCHAR(400),
    DETAILED_STATUS VARCHAR(400),
    CATEGORY VARCHAR(400),
    LOCATION_NAME VARCHAR(400),
    FACILITY_INFO VARCHAR(400),
    ROOM_DETAILS VARCHAR(400),
    REASON_DESCRIPTION VARCHAR(400),
    STAFF_NAME VARCHAR(400),
    LAST_MODIFIED_DATETIME TIMESTAMP,
    ACTIVE_FLAG VARCHAR(10),
    DATA_SOURCE_CODE VARCHAR(200)
);

-- Insert sample data for testing
INSERT INTO HEALTHDB.MEMBER_APPOINTMENTS VALUES
('SCH001', 'MBR12345', 'APPT_001_001', 'GRP_001', 1, TRUE, 'Oncology Patient', 
 '2025-01-15 09:00:00', '2025-01-15 10:30:00', 90, 'Scheduled', 'Confirmed by patient',
 'Consultation', 'Moffitt Main Campus', 'Building A - Cancer Center', 'Room 201',
 'Initial consultation for breast cancer', 'Dr. Sarah Johnson', '2025-01-10 14:30:00', 'TRUE', 'EPIC'),

('SCH002', 'MBR67890', 'APPT_002_001', 'GRP_001', 1, TRUE, 'Oncology Patient',
 '2025-01-15 14:00:00', '2025-01-15 15:30:00', 90, 'Completed', 'Patient attended',
 'Follow-up', 'Moffitt Main Campus', 'Building A - Cancer Center', 'Room 150',
 'Post-treatment follow-up', 'Dr. Emily Rodriguez', '2025-01-15 15:45:00', 'TRUE', 'EPIC'),

('SCH003', 'MBR24681', 'APPT_003_001', 'GRP_001', 2, TRUE, 'Oncology Patient',
 '2025-01-16 11:00:00', '2025-01-16 12:00:00', 60, 'Scheduled', 'Insurance pre-auth pending',
 'Treatment', 'Moffitt Main Campus', 'Building A - Infusion Center', 'Chair 12',
 'Chemotherapy infusion', 'Nurse practitioner Lisa Wong', '2025-01-13 11:45:00', 'TRUE', 'EPIC'),

('SCH004', 'MBR13579', 'APPT_004_001', 'GRP_004', 1, TRUE, 'Oncology Patient',
 '2025-01-16 13:30:00', '2025-01-16 14:15:00', 45, 'No Show', 'Patient did not attend',
 'Consultation', 'Moffitt Satellite Clinic', 'Building D - Outpatient', 'Room 45',
 'Second opinion consultation', 'Dr. Amanda Foster', '2025-01-16 14:30:00', 'TRUE', 'EPIC'),

('SCH005', 'MBR98765', 'APPT_005_001', 'GRP_001', 1, TRUE, 'Oncology Patient',
 '2025-01-17 15:00:00', '2025-01-17 16:30:00', 90, 'Rescheduled', 'Moved to next week',
 'Consultation', 'Moffitt Main Campus', 'Building A - Cancer Center', 'Room 180',
 'Surgical consultation', 'Dr. Thomas Wilson', '2025-01-16 13:15:00', 'TRUE', 'EPIC');

-- Supporting tables for the view
CREATE OR REPLACE TABLE HEALTHDB.MEMBER_INFO (
    MEMBER_ID VARCHAR(200),
    ANONYMIZED_MEMBER_ID VARCHAR(200),
    MEMBER_NAME VARCHAR(400),
    DATE_OF_BIRTH DATE,
    PHONE_NUMBER VARCHAR(20)
);

INSERT INTO HEALTHDB.MEMBER_INFO VALUES
('MBR12345', 'ANON_001', 'John Smith', '1980-05-15', '555-0101'),
('MBR67890', 'ANON_002', 'Jane Doe', '1975-08-22', '555-0102'),
('MBR24681', 'ANON_003', 'Robert Johnson', '1965-12-10', '555-0103'),
('MBR13579', 'ANON_004', 'Maria Garcia', '1990-03-18', '555-0104'),
('MBR98765', 'ANON_005', 'David Wilson', '1955-11-30', '555-0105');

-- Configuration tables
CREATE OR REPLACE TABLE CONFIGURATION.TIME_ADJUSTMENTS (
    MEMBER_ID VARCHAR(200),
    TIME_ADJUSTMENT INTEGER -- hours to add/subtract for privacy
);

INSERT INTO CONFIGURATION.TIME_ADJUSTMENTS VALUES
('MBR12345', 2),
('MBR67890', -1),
('MBR24681', 3),
('MBR13579', -2),
('MBR98765', 1);

CREATE OR REPLACE TABLE CONFIGURATION.ACCESS_CONTROL_RULES (
    ROLE_NAME VARCHAR(200),
    ACCESS_GRANTED BOOLEAN,
    DESCRIPTION VARCHAR(400)
);

INSERT INTO CONFIGURATION.ACCESS_CONTROL_RULES VALUES
('DOCTOR', TRUE, 'Full access to patient data'),
('NURSE', TRUE, 'Full access to patient data'),
('RESEARCHER', FALSE, 'Anonymized access only'),
('ADMINISTRATOR', FALSE, 'Limited access for admin tasks'),
('ACCOUNTADMIN', TRUE, 'System admin access');