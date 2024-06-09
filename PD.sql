-- MySQL dump 10.13  Distrib 8.0.19, for Win64 (x86_64)
--
-- Host: 127.0.0.1    Database: pd
-- ------------------------------------------------------
-- Server version	8.0.19

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `account`
--

DROP TABLE IF EXISTS `account`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `account` (
  `username` varchar(40) NOT NULL,
  `Name` varchar(45) NOT NULL,
  `Email` varchar(255) NOT NULL,
  `TempPassFlag` enum('0','1') DEFAULT '0',
  `Password` varchar(255) NOT NULL,
  `Role` varchar(45) NOT NULL,
  `TempPassDate` date DEFAULT NULL,
  `EmailConfirmationSentOn` datetime DEFAULT NULL,
  `EmailConfirmed` enum('0','1') DEFAULT '0',
  `ConfirmedEmailOn` datetime DEFAULT NULL,
  PRIMARY KEY (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `account`
--

LOCK TABLES `account` WRITE;
/*!40000 ALTER TABLE `account` DISABLE KEYS */;
INSERT INTO `account` VALUES ('admin','admin','admin@admin.com','0','$5$rounds=535000$BLgG/fb.eWPgizEp$onHlEfsC8EjsuXMiJuSmXa9bN7LrAF0syeGRPasQ5x.','admin',NULL,NULL,'0',NULL);
/*!40000 ALTER TABLE `account` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `model`
--

DROP TABLE IF EXISTS `model`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `model` (
  `ModelID` int NOT NULL AUTO_INCREMENT,
  `ModelName` varchar(45) NOT NULL,
  `DiseaseType` enum('CKD','Diabetes','CHD','RA','Schizophrenia','TC','Asthma','Alzheimer','Hypothyroidism','Breast Cancer','ADHD','PC','MS','Lung Cancer','Glaucoma') NOT NULL,
  `Accuracy` decimal(12,5) NOT NULL,
  `Active` enum('0','1') NOT NULL,
  `TotalInstances` int DEFAULT NULL,
  `TestInstances` int DEFAULT NULL,
  `TrainingPercent` decimal(5,2) DEFAULT NULL,
  PRIMARY KEY (`ModelID`)
) ENGINE=InnoDB AUTO_INCREMENT=38 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `model`
--

LOCK TABLES `model` WRITE;
/*!40000 ALTER TABLE `model` DISABLE KEYS */;
INSERT INTO `model` VALUES (1,'2022-04-30 01.07.18.sav','Diabetes',0.87500,'1',399,80,80.00),(2,'2022-04-29 00.32.18.sav','CKD',0.97959,'1',244,49,80.00),(3,'2022-04-29 00.41.03.sav','CHD',0.93076,'1',1300,130,90.00),(4,'2022-04-29 00.41.30.sav','RA',0.92310,'1',1300,130,90.00),(5,'2022-04-29 00.42.10.sav','Schizophrenia',0.63636,'1',219,44,80.00),(6,'2022-04-29 00.42.52.sav','TC',0.90909,'1',536,54,90.00),(7,'2022-04-29 00.43.40.sav','Asthma',0.93939,'1',328,66,80.00),(8,'Alzheimer.sav','Alzheimer',0.95555,'1',151,45,93.00),(9,'hypovoting947368.sav', 'Hypothyroidism',0.947368,'1',150,45, 1.00),(10,'PC.sav', 'PC',0.9,'1',90,11, 90.0),(11,'MS.sav', 'MS',0.87,'1',87,14, 87.0),(12,'glaucoma.sav','Glaucoma',0.88760,'1',196,20,90.00),(13,'lung.sav','Lung Cancer',0.96290,'1',309,31,90.00);

;
/*!40000 ALTER TABLE `model` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `patient`
--

DROP TABLE IF EXISTS `patient`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `patient` (
  `NationalID` varchar(10) NOT NULL,
  `Name` varchar(45) NOT NULL,
  `BirthDate` date NOT NULL,
  `Gender` enum('Male','Female') NOT NULL,
  PRIMARY KEY (`NationalID`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `patient`
--

LOCK TABLES `patient` WRITE;
/*!40000 ALTER TABLE `patient` DISABLE KEYS */;
/*!40000 ALTER TABLE `patient` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `result`
--

DROP TABLE IF EXISTS `result`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `result` (
  `ResultID` int NOT NULL AUTO_INCREMENT,
  `date` date NOT NULL,
  `TestResult` enum('Positive','Negative') NOT NULL,
  `NationalID` varchar(10) NOT NULL,
  `ModelID` int NOT NULL,
  PRIMARY KEY (`ResultID`),
  KEY `NationalID_idx` (`NationalID`),
  CONSTRAINT `NationalID` FOREIGN KEY (`NationalID`) REFERENCES `patient` (`NationalID`)
) ENGINE=InnoDB AUTO_INCREMENT=14 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `result`
--

LOCK TABLES `result` WRITE;
/*!40000 ALTER TABLE `result` DISABLE KEYS */;
/*!40000 ALTER TABLE `result` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `tempmodel`
--

DROP TABLE IF EXISTS `tempmodel`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `tempmodel` (
  `ModelID` int NOT NULL AUTO_INCREMENT,
  `ModelName` varchar(45) NOT NULL,
  `ModelType` enum('CKD','Diabetes','CHD','RA','Schizophrenia','TC','Asthma','Alzheimer','Hypothyroidism','Breast Cancer','ADHD','PC','MS','Lung Cancer','Glaucoma') NOT NULL,
  `TrainingPercent` decimal(5,2) NOT NULL,
  `Accuracy` decimal(12,5) NOT NULL,
  `TestInstances` int DEFAULT NULL,
  `TotalInstances` int DEFAULT NULL,
  PRIMARY KEY (`ModelID`)
) ENGINE=InnoDB AUTO_INCREMENT=25 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `tempmodel`
--

LOCK TABLES `tempmodel` WRITE;
/*!40000 ALTER TABLE `tempmodel` DISABLE KEYS */;
/*!40000 ALTER TABLE `tempmodel` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-04-26 23:24:24
