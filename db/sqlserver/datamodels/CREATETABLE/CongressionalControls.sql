CREATE TABLE CongressionalControls
(
	CongressionalControlsId  INT           NOT NULL UNIQUE,
	FundCode                 NVARCHAR(150) NULL DEFAULT ('NS'),
	FundName                 NVARCHAR(150) NULL DEFAULT ('NS'),
	ProgramAreaCode          NVARCHAR(150) NULL DEFAULT ('NS'),
	ProgramAreaName          NVARCHAR(150) NULL DEFAULT ('NS'),
	ProgramProjectCode       NVARCHAR(150) NULL DEFAULT ('NS'),
	ProgramProjectName       NVARCHAR(150) NULL DEFAULT ('NS'),
	SubProjectCode           NVARCHAR(150) NULL DEFAULT ('NS'),
	SubProjectName           NVARCHAR(150) NULL DEFAULT ('NS'),
	ReprogrammingRestriction NVARCHAR(150) NULL DEFAULT ('NS'),
	IncreaseRestriction      NVARCHAR(150) NULL DEFAULT ('NS'),
	DecreaseRestriction      NVARCHAR(150) NULL DEFAULT ('NS'),
	MemoRequirement          NVARCHAR(150) NULL DEFAULT ('NS'),
	CONSTRAINT CongressionalControlsPrimaryKey PRIMARY KEY
		(
		  CongressionalControlsId ASC
			)
);
