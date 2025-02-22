/****** Object:  Table StatusOfSpecialAccountFunds    Script Date: 10/7/2023 1:42:13 PM ******/

CREATE TABLE StatusOfSpecialAccountFunds
(
	StatusOfSpecialAccountFundsId INT           NOT NULL UNIQUE,
	Fiscal                        Year NVARCHAR(255) NULL,
	BFY                           NVARCHAR(255) NULL,
	EFY                           NVARCHAR(255) NULL,
	FundCode                      NVARCHAR(255) NULL,
	FundName                      NVARCHAR(255) NULL,
	RpioCode                      NVARCHAR(255) NULL,
	RpioName                      NVARCHAR(255) NULL,
	ProgramCode                   NVARCHAR(255) NULL,
	SpecialAccountNumber          NVARCHAR(255) NULL,
	SpecialAccountName            NVARCHAR(255) NULL,
	SpecialAccountStatus          NVARCHAR(255) NULL,
	NplStatusCode                 NVARCHAR(255) NULL,
	StatusDescription             NVARCHAR(255) NULL,
	EpaSiteId                     NVARCHAR(255) NULL,
	CerclisSiteId                 NVARCHAR(255) NULL,
	SiteCode                      NVARCHAR(255) NULL,
	SiteName                      NVARCHAR(255) NULL,
	OperableUnit                  NVARCHAR(255) NULL,
	PipelineCode                  NVARCHAR(255) NULL,
	PipelineDescription           NVARCHAR(255) NULL,
	TransactionDescription        NVARCHAR(255) NULL,
	InterestDate                  NVARCHAR(255) NULL,
	TrustfundTransfers            INT           NULL,
	OpenCommitments               FLOAT         NULL DEFAULT (0.0),
	Obligations                   FLOAT         NULL DEFAULT (0.0),
	UnliquidatedObligations       FLOAT         NULL DEFAULT (0.0),
	Disbursements                 FLOAT         NULL DEFAULT (0.0),
	UnpaidBalance                 INT           NULL,
	CumulativeReceipts            FLOAT         NULL DEFAULT (0.0),
	NetReceipts                   INT           NULL,
	Interest                      INT           NULL,
	CollectionsAndInterest        FLOAT         NULL DEFAULT (0.0),
	AvailableBalance              float         NULL,
	CONSTRAINT StatufOfSpecialAccountFundsPrimaryKey PRIMARY KEY
		(
		  StatusOfSpecialAccountFundsId ASC
			)
)
