CREATE TABLE IF NOT EXISTS "Regulations" 
(
	"RegulationsId"	INTEGER NOT NULL UNIQUE,
	"RegulationName"	TEXT(80) DEFAULT 'NS',
	"Document"	TEXT(80) DEFAULT 'NS',
	PRIMARY KEY("RegulationsId" AUTOINCREMENT)
);
