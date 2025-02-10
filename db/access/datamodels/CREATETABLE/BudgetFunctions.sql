CREATE TABLE IF NOT EXISTS "BudgetFunctions" 
(
	"BudgetFunctionsId"	INTEGER NOT NULL UNIQUE,
	"FunctionCode"	TEXT(80) DEFAULT 'NS',
	"FunctionName"	TEXT(80) DEFAULT 'NS',
	"SubFunctionCode"	TEXT(80) DEFAULT 'NS',
	"SubFunctionName"	TEXT(80) DEFAULT 'NS',
	PRIMARY KEY("BudgetFunctionsId" AUTOINCREMENT)
);
