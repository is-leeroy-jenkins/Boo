CREATE TABLE FundSymbols
(
	FundSymbolsId            INT           NOT NULL UNIQUE,
	BFY                      NVARCHAR(150) NULL DEFAULT ('NS'),
	EFY                      NVARCHAR(150) NULL DEFAULT ('NS'),
	FundCode                 NVARCHAR(150) NULL DEFAULT ('NS'),
	FundName                 NVARCHAR(150) NULL DEFAULT ('NS'),
	TreasuryAccountCode      NVARCHAR(150) NULL DEFAULT ('NS'),
	TreasuryAccountName      NVARCHAR(150) NULL DEFAULT ('NS'),
	BudgetAccountCode        NVARCHAR(150) NULL DEFAULT ('NS'),
	BudgetAccountName        NVARCHAR(150) NULL DEFAULT ('NS'),
	ApportionmentAccountCode NVARCHAR(150) NULL DEFAULT ('NS'),
	CONSTRAINT FundSymbolsPrimaryKey PRIMARY KEY
		(
		  FundSymbolsId ASC
			)
);
