CREATE TABLE IF NOT EXISTS "BudgetObjectClasses" 
(
	"BudgetObjectClassesId"	INTEGER NOT NULL UNIQUE,
	"Code"	TEXT(80) DEFAULT 'NS',
	"Name"	TEXT(80) DEFAULT 'NS',
	PRIMARY KEY("BudgetObjectClassesId" AUTOINCREMENT)
);
