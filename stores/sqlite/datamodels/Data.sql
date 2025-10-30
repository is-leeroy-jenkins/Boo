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

CREATE TABLE IF NOT EXISTS "BudgetaryResourceExecution" 
(
	"BudgetaryResourceExecutionId"	INTEGER,
	"ReportYear"	TEXT(80) DEFAULT 'NS',
	"BeginningPeriodOfAvailability"	TEXT(80) DEFAULT 'NS',
	"EndingPeriodOfAvailability"	TEXT(80) DEFAULT 'NS',
	"Agency"	TEXT(80) DEFAULT 'NS',
	"Bureau"	TEXT(80) DEFAULT 'NS',
	"BudgetAgencyCode"	TEXT(80) DEFAULT 'NS',
	"BudgetBureauCode"	TEXT(80) DEFAULT 'NS',
	"BudgetAccountCode"	TEXT(80) DEFAULT 'NS',
	"BudgetAccountName"	TEXT(80) DEFAULT 'NS',
	"TreasuryAgencyCode"	TEXT(80) DEFAULT 'NS',
	"AllocationAccount"	TEXT(80) DEFAULT 'NS',
	"TreasuryAccountCode"	TEXT(80) DEFAULT 'NS',
	"TreasuryAccountName"	TEXT(80) DEFAULT 'NS',
	"TreasuryFundSymbol"	TEXT(80) DEFAULT 'NS',
	"LineNumber"	TEXT(80) DEFAULT 'NS',
	"LineDescription"	TEXT(80) DEFAULT 'NS',
	"SectionName"	TEXT(80) DEFAULT 'NS',
	"SectionNumber"	TEXT(80) DEFAULT 'NS',
	"LineType"	TEXT(80) DEFAULT 'NS',
	"Updated"	TEXT(80) DEFAULT 'NS',
	"November"	REAL DEFAULT 0.00,
	"December"	REAL DEFAULT 0.00,
	"January"	REAL DEFAULT 0.00,
	"Feburary"	REAL DEFAULT 0.00,
	"March"	REAL DEFAULT 0.00,
	"April"	REAL DEFAULT 0.00,
	"May"	REAL DEFAULT 0.00,
	"June"	REAL DEFAULT 0.00,
	"July"	REAL DEFAULT 0.00,
	"August"	REAL DEFAULT 0.00,
	"September"	REAL DEFAULT 0.00,
	"October"	REAL DEFAULT 0.00
);

CREATE TABLE IF NOT EXISTS "BudgetFunctions" 
(
	"BudgetFunctionsId"	INTEGER NOT NULL UNIQUE,
	"FunctionCode"	TEXT(80) DEFAULT 'NS',
	"FunctionName"	TEXT(80) DEFAULT 'NS',
	"SubFunctionCode"	TEXT(80) DEFAULT 'NS',
	"SubFunctionName"	TEXT(80) DEFAULT 'NS',
	PRIMARY KEY("BudgetFunctionsId" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "BudgetObjectClasses" 
(
	"BudgetObjectClassesId"	INTEGER NOT NULL UNIQUE,
	"Code"	TEXT(80) DEFAULT 'NS',
	"Name"	TEXT(80) DEFAULT 'NS',
	PRIMARY KEY("BudgetObjectClassesId" AUTOINCREMENT)
)


CREATE TABLE IF NOT EXISTS "ChatLog" 
(
	"ID"	INTEGER NOT NULL UNIQUE,
	"Time"	TEXT(80) DEFAULT 'NS',
	"Role"	TEXT(80) DEFAULT 'NS',
	"Content"	TEXT(80) DEFAULT 'NS',
	CONSTRAINT "ChatLogPrimaryKey" PRIMARY KEY("ID" AUTOINCREMENT)
);

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

CREATE TABLE IF NOT EXISTS "FiscalYears" 
(
	"FiscalYearsId"	INTEGER NOT NULL UNIQUE,
	"BFY"	TEXT(80) DEFAULT 'NS',
	"EFY"	TEXT(80) DEFAULT 'NS',
	"StartDate"	TEXT(80) DEFAULT 'NS',
	"ColumbusDay"	TEXT(80) DEFAULT 'NS',
	"VeteransDay"	TEXT(80) DEFAULT 'NS',
	"ThanksgivingDay"	TEXT(80) DEFAULT 'NS',
	"ChristmasDay"	TEXT(80) DEFAULT 'NS',
	"NewYearsDay"	TEXT(80) DEFAULT 'NS',
	"MartinLutherKingsDay"	TEXT(80) DEFAULT 'NS',
	"WashingtonsDay"	TEXT(80) DEFAULT 'NS',
	"MemorialDay"	TEXT(80) DEFAULT 'NS',
	"JuneteenthDay"	TEXT(80) DEFAULT 'NS',
	"IndependenceDay"	TEXT(80) DEFAULT 'NS',
	"LaborDay"	TEXT(80) DEFAULT 'NS',
	"ExpiringYear"	TEXT(80) DEFAULT 'NS',
	"ExpirationDate"	TEXT(80) DEFAULT 'NS',
	"WorkDays"	INTEGER,
	"WeekDays"	INTEGER,
	"WeekEnds"	INTEGER,
	"EndDate"	TEXT(80) DEFAULT 'NS',
	"Availability"	TEXT(80) DEFAULT 'NS',
	PRIMARY KEY("FiscalYearsId" AUTOINCREMENT)
);


CREATE TABLE IF NOT EXISTS "MainAccounts" 
(
	"MainAccountsId"	INTEGER NOT NULL UNIQUE,
	"AgencyIdentifier"	TEXT(80) DEFAULT 'NS',
	"AgencyCode"	TEXT(80) DEFAULT 'NS',
	"Code"	TEXT(80) DEFAULT 'NS',
	"Name"	TEXT(80) DEFAULT 'NS',
	"Type"	TEXT(80) DEFAULT 'NS',
	PRIMARY KEY("MainAccountsId" AUTOINCREMENT)
);


CREATE TABLE IF NOT EXISTS "ProductCategories" 
(
	"ProductCategoriesId"	INTEGER NOT NULL UNIQUE,
	"Category"	TEXT(80) DEFAULT 'NS',
	"Name"	TEXT(80) DEFAULT 'NS',
	"Manager"	TEXT(80) DEFAULT 'NS',
	PRIMARY KEY("ProductCategoriesId" AUTOINCREMENT)
);


CREATE TABLE IF NOT EXISTS "Regulations" 
(
	"RegulationsId"	INTEGER NOT NULL UNIQUE,
	"RegulationName"	TEXT(80) DEFAULT 'NS',
	"Document"	TEXT(80) DEFAULT 'NS',
	PRIMARY KEY("RegulationsId" AUTOINCREMENT)
);



CREATE TABLE IF NOT EXISTS "ResourceLines" 
(
	"ReportingLinesId"	INTEGER NOT NULL UNIQUE,
	"Number"	TEXT(80) DEFAULT 'NS',
	"Name"	TEXT(80) DEFAULT 'NS',
	"Caption"	TEXT(80) DEFAULT 'NS',
	"Category"	TEXT(80) DEFAULT 'NS',
	"Range"	TEXT(80) DEFAULT 'NS',
	PRIMARY KEY("ReportingLinesId" AUTOINCREMENT)
);



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