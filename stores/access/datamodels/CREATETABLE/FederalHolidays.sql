CREATE TABLE IF NOT EXISTS "FederalHolidays" 
(
	"FederalHolidaysId"	INTEGER NOT NULL UNIQUE,
	"BFY"	TEXT(80) DEFAULT 'NS',
	"ColumbusDay"	TEXT(80) DEFAULT 'NS',
	"VeteransDay"	TEXT(80) DEFAULT 'NS',
	"ThanksgivingDay"	TEXT(80) DEFAULT 'NS',
	"ChristmasDay"	TEXT(80) DEFAULT 'NS',
	"NewYearsDay"	TEXT(80) DEFAULT 'NS',
	"MartinLutherKingDay"	TEXT(80) DEFAULT 'NS',
	"PresidentsDay"	TEXT(80) DEFAULT 'NS',
	"MemorialDay"	TEXT(80) DEFAULT 'NS',
	"JuneteenthDay"	TEXT(80) DEFAULT 'NS',
	"IndependenceDay"	TEXT(80) DEFAULT 'NS',
	"LaborDay"	TEXT(80) DEFAULT 'NS',
	PRIMARY KEY("FederalHolidaysId" AUTOINCREMENT)
);