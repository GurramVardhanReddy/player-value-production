# Snowflake schema for published scores, in the MARKETING database created by
# github.com/GurramVardhanReddy/marketing-data-platform (infra/snowflake). The TRANSFORMER role
# from that stack writes here.
terraform {
  required_version = ">= 1.6"
  required_providers {
    snowflake = {
      source  = "snowflakedb/snowflake"
      version = "~> 2.3"
    }
  }
}

# Credentials from the environment (SNOWFLAKE_ORGANIZATION_NAME, SNOWFLAKE_ACCOUNT_NAME,
# SNOWFLAKE_USER, SNOWFLAKE_AUTHENTICATOR=SNOWFLAKE_JWT, SNOWFLAKE_PRIVATE_KEY).
provider "snowflake" {
  role = "ACCOUNTADMIN"
}

variable "database" {
  type    = string
  default = "MARKETING"
}

variable "writer_role" {
  type    = string
  default = "TRANSFORMER"
}

resource "snowflake_schema" "player_value" {
  database = var.database
  name     = "PLAYER_VALUE"
  comment  = "Weekly player value scores published by player-value-production"
}

resource "snowflake_grant_privileges_to_account_role" "writer" {
  account_role_name = var.writer_role
  privileges        = ["USAGE", "CREATE TABLE"]
  on_schema {
    schema_name = "\"${var.database}\".\"${snowflake_schema.player_value.name}\""
  }
}

output "schema" {
  value = "${var.database}.${snowflake_schema.player_value.name}"
}
