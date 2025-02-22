CREATE TABLE AppropriationLevelAuthority
(
	AppropriationLevelAuthorityId INT           NOT NULL UNIQUE,
	BFY                           NVARCHAR(150) NULL DEFAULT ('NS'),
	EFY                           NVARCHAR(150) NULL DEFAULT ('NS'),
	TreasuryAccountCode           NVARCHAR(150) NULL DEFAULT ('NS'),
	FundCode                      NVARCHAR(150) NULL DEFAULT ('NS'),
	FundName                      NVARCHAR(150) NULL DEFAULT ('NS'),
	BudgetLevel                   NVARCHAR(150) NULL DEFAULT ('NS'),
	Budgeted                      FLOAT         NULL DEFAULT (0.0),
	Posted                        FLOAT         NULL DEFAULT (0.0),
	CarryoverOut                  FLOAT         NULL DEFAULT (0.0),
	CarryoverIn                   FLOAT         NULL DEFAULT (0.0),
	Reimbursements                FLOAT         NULL DEFAULT (0.0),
	Recoveries                    FLOAT         NULL DEFAULT (0.0),
	TreasuryAccountName           NVARCHAR(150) NULL DEFAULT ('NS'),
	BudgetAccountCode             NVARCHAR(150) NULL DEFAULT ('NS'),
	BudgetAccountName             NVARCHAR(150) NULL DEFAULT ('NS'),
	CONSTRAINT AppropriationLevelAuthorityPrimaryKey PRIMARY KEY
		(
		  AppropriationLevelAuthorityId ASC
			)
);
