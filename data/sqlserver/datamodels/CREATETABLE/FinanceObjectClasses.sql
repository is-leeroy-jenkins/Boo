CREATE TABLE FinanceObjectClasses
(
	FinanceObjectClassesId INT           NOT NULL UNIQUE,
	Code                   NVARCHAR(150) NULL DEFAULT ('NS'),
	Name                   NVARCHAR(150) NULL DEFAULT ('NS'),
	BocCode                NVARCHAR(150) NULL DEFAULT ('NS'),
	BocName                NVARCHAR(150) NULL DEFAULT ('NS'),
	CONSTRAINT FinanceObjectClassesPrimaryKey PRIMARY KEY
		(
		  FinanceObjectClassesId ASC
			)
);
