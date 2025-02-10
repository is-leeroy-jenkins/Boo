CREATE TABLE IF NOT EXISTS "ProductCategories" 
(
	"ProductCategoriesId"	INTEGER NOT NULL UNIQUE,
	"Category"	TEXT(80) DEFAULT 'NS',
	"Name"	TEXT(80) DEFAULT 'NS',
	"Manager"	TEXT(80) DEFAULT 'NS',
	PRIMARY KEY("ProductCategoriesId" AUTOINCREMENT)
);