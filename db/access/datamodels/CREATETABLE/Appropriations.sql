CREATE TABLE IF NOT EXISTS "Appropriations" 
(
	"AppropriationsId"	INTEGER NOT NULL UNIQUE,
	"FiscalYear"	TEXT(80) DEFAULT 'NS',
	"PublicLaw"	TEXT(80) DEFAULT 'NS',
	"AppropriationTitle"	TEXT(80) DEFAULT 'NS',
	"EnactedDate"	TEXT(80) DEFAULT 'NS',
	"ExplanatoryComments"	TEXT(80) DEFAULT 'NS',
	"Authority"	TEXT(80) DEFAULT 'NS',
	PRIMARY KEY("AppropriationsId" AUTOINCREMENT)
);
