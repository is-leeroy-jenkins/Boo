CREATE TABLE AggregateOutlays
(
	MonthlyAggregatesId INT           NOT NULL UNIQUE,
	BudgetAccountName   NVARCHAR(255) NULL,
	MainAccount         NVARCHAR(255) NULL,
	October             FLOAT         NULL DEFAULT (0.0),
	November            FLOAT         NULL DEFAULT (0.0),
	December            FLOAT         NULL DEFAULT (0.0),
	January             FLOAT         NULL DEFAULT (0.0),
	Feburary            FLOAT         NULL DEFAULT (0.0),
	March               FLOAT         NULL DEFAULT (0.0),
	April               FLOAT         NULL DEFAULT (0.0),
	May                 FLOAT         NULL DEFAULT (0.0),
	June                FLOAT         NULL DEFAULT (0.0),
	July                FLOAT         NULL DEFAULT (0.0),
	August              FLOAT         NULL DEFAULT (0.0),
	September           FLOAT         NULL DEFAULT (0.0),
	CONSTRAINT AggregateOutlaysPrimaryKey PRIMARY KEY
		(
		  MonthlyAggregatesId ASC
			)
);
