CREATE TABLE IF NOT EXISTS "TreasurySymbols" 
(
	"TreasurySymbolId"	INTEGER NOT NULL UNIQUE,
	"AgencyIdentifier"	TEXT(80) DEFAULT 'NS',
	"MainAccount"	TEXT(80) DEFAULT 'NS',
	"Availability"	TEXT(80) DEFAULT 'NS',
	"TreasuryAccount"	TEXT(80) DEFAULT 'NS',
	"Agency"	TEXT(80) DEFAULT 'NS',
	"Title"	TEXT(80) DEFAULT 'NS',
	"Legislation"	TEXT(80) DEFAULT 'NS',
	"FundType"	TEXT(80) DEFAULT 'NS',
	PRIMARY KEY("TreasurySymbolId" AUTOINCREMENT)
);
