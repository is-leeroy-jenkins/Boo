CREATE TABLE DataRuleDescriptions
(
	DataRuleDescriptionsId INT           NOT NULL UNIQUE,
	Schedule               NVARCHAR(150) NULL DEFAULT ('NS'),
	LineNumber             NVARCHAR(150) NULL DEFAULT ('NS'),
	RuleNumber             NVARCHAR(150) NULL DEFAULT ('NS'),
	RuleDescription        NVARCHAR(MAX) NULL,
	ScheduleOrder          NVARCHAR(150) NULL DEFAULT ('NS'),
	CONSTRAINT DataRuleDescriptionsPrimaryKey PRIMARY KEY
		(
		  DataRuleDescriptionsId ASC
			)
);
