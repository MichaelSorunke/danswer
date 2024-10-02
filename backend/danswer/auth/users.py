import smtplib
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from datetime import timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional
from typing import Tuple

import jwt
from email_validator import EmailNotValidError
from email_validator import validate_email
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi import status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi_users import BaseUserManager
from fastapi_users import exceptions
from fastapi_users import FastAPIUsers
from fastapi_users import models
from fastapi_users import schemas
from fastapi_users import UUIDIDMixin
from fastapi_users.authentication import AuthenticationBackend
from fastapi_users.authentication import CookieTransport
from fastapi_users.authentication import JWTStrategy
from fastapi_users.authentication import Strategy
from fastapi_users.authentication.strategy.db import AccessTokenDatabase
from fastapi_users.authentication.strategy.db import DatabaseStrategy
from fastapi_users.openapi import OpenAPIResponseType
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from sqlalchemy import select
from sqlalchemy.orm import attributes
from sqlalchemy.orm import Session

from danswer.auth.invited_users import get_invited_users
from danswer.auth.schemas import UserCreate
from danswer.auth.schemas import UserRole
from danswer.auth.schemas import UserUpdate
from danswer.configs.app_configs import AUTH_TYPE
from danswer.configs.app_configs import DATA_PLANE_SECRET
from danswer.configs.app_configs import DISABLE_AUTH
from danswer.configs.app_configs import EMAIL_FROM
from danswer.configs.app_configs import EXPECTED_API_KEY
from danswer.configs.app_configs import MULTI_TENANT
from danswer.configs.app_configs import REQUIRE_EMAIL_VERIFICATION
from danswer.configs.app_configs import SECRET_JWT_KEY
from danswer.configs.app_configs import SESSION_EXPIRE_TIME_SECONDS
from danswer.configs.app_configs import SMTP_PASS
from danswer.configs.app_configs import SMTP_PORT
from danswer.configs.app_configs import SMTP_SERVER
from danswer.configs.app_configs import SMTP_USER
from danswer.configs.app_configs import TRACK_EXTERNAL_IDP_EXPIRY
from danswer.configs.app_configs import USER_AUTH_SECRET
from danswer.configs.app_configs import VALID_EMAIL_DOMAINS
from danswer.configs.app_configs import WEB_DOMAIN
from danswer.configs.constants import AuthType
from danswer.configs.constants import DANSWER_API_KEY_DUMMY_EMAIL_DOMAIN
from danswer.configs.constants import DANSWER_API_KEY_PREFIX
from danswer.configs.constants import UNNAMED_KEY_PLACEHOLDER
from danswer.db.auth import get_access_token_db
from danswer.db.auth import get_default_admin_user_emails
from danswer.db.auth import get_user_count
from danswer.db.auth import get_user_db
from danswer.db.auth import SQLAlchemyUserAdminDB
from danswer.db.engine import get_async_session_with_tenant
from danswer.db.engine import get_session
from danswer.db.engine import get_session_with_tenant
from danswer.db.engine import get_sqlalchemy_engine
from danswer.db.models import AccessToken
from danswer.db.models import OAuthAccount
from danswer.db.models import User
from danswer.db.models import UserTenantMapping
from danswer.db.users import get_user_by_email
from danswer.utils.logger import setup_logger
from danswer.utils.telemetry import optional_telemetry
from danswer.utils.telemetry import RecordType
from danswer.utils.variable_functionality import fetch_versioned_implementation
from shared_configs.configs import current_tenant_id

logger = setup_logger()


def is_user_admin(user: User | None) -> bool:
    if AUTH_TYPE == AuthType.DISABLED:
        return True
    if user and user.role == UserRole.ADMIN:
        return True
    return False


def verify_auth_setting() -> None:
    if AUTH_TYPE not in [AuthType.DISABLED, AuthType.BASIC, AuthType.GOOGLE_OAUTH]:
        raise ValueError(
            "User must choose a valid user authentication method: "
            "disabled, basic, or google_oauth"
        )
    logger.notice(f"Using Auth Type: {AUTH_TYPE.value}")


def get_display_email(email: str | None, space_less: bool = False) -> str:
    if email and email.endswith(DANSWER_API_KEY_DUMMY_EMAIL_DOMAIN):
        name = email.split("@")[0]
        if name == DANSWER_API_KEY_PREFIX + UNNAMED_KEY_PLACEHOLDER:
            return "Unnamed API Key"

        if space_less:
            return name

        return name.replace("API_KEY__", "API Key: ")

    return email or ""


def user_needs_to_be_verified() -> bool:
    # all other auth types besides basic should require users to be
    # verified
    return AUTH_TYPE != AuthType.BASIC or REQUIRE_EMAIL_VERIFICATION


def verify_email_is_invited(email: str) -> None:
    whitelist = get_invited_users()
    if not whitelist:
        return

    if not email:
        raise PermissionError("Email must be specified")

    email_info = validate_email(email)  # can raise EmailNotValidError

    for email_whitelist in whitelist:
        try:
            # normalized emails are now being inserted into the db
            # we can remove this normalization on read after some time has passed
            email_info_whitelist = validate_email(email_whitelist)
        except EmailNotValidError:
            continue

        # oddly, normalization does not include lowercasing the user part of the
        # email address ... which we want to allow
        if email_info.normalized.lower() == email_info_whitelist.normalized.lower():
            return

    raise PermissionError("User not on allowed user whitelist")


def verify_email_in_whitelist(email: str, tenant_id: str | None = None) -> None:
    with get_session_with_tenant(tenant_id) as db_session:
        if not get_user_by_email(email, db_session):
            verify_email_is_invited(email)


def verify_email_domain(email: str) -> None:
    if VALID_EMAIL_DOMAINS:
        if email.count("@") != 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email is not valid",
            )
        domain = email.split("@")[-1]
        if domain not in VALID_EMAIL_DOMAINS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email domain is not valid",
            )


def get_tenant_id_for_email(email: str) -> str:
    # Implement logic to get tenant_id from the mapping table
    with Session(get_sqlalchemy_engine()) as db_session:
        result = db_session.execute(
            select(UserTenantMapping.tenant_id).where(UserTenantMapping.email == email)
        )
        tenant_id = result.scalar_one_or_none()
    if tenant_id is None:
        raise exceptions.UserNotExists()
    return tenant_id


def send_user_verification_email(
    user_email: str,
    token: str,
    mail_from: str = EMAIL_FROM,
) -> None:
    msg = MIMEMultipart()
    msg["Subject"] = "Danswer Email Verification"
    msg["To"] = user_email
    if mail_from:
        msg["From"] = mail_from

    link = f"{WEB_DOMAIN}/auth/verify-email?token={token}"

    body = MIMEText(f"Click the following link to verify your email address: {link}")
    msg.attach(body)

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
        s.starttls()
        # If credentials fails with gmail, check (You need an app password, not just the basic email password)
        # https://support.google.com/accounts/answer/185833?sjid=8512343437447396151-NA
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = USER_AUTH_SECRET
    verification_token_secret = USER_AUTH_SECRET

    async def create(
        self,
        user_create: schemas.UC | UserCreate,
        safe: bool = False,
        request: Optional[Request] = None,
    ) -> User:
        verify_email_is_invited(user_create.email)
        verify_email_domain(user_create.email)
        if hasattr(user_create, "role"):
            user_count = await get_user_count()
            if user_count == 0 or user_create.email in get_default_admin_user_emails():
                user_create.role = UserRole.ADMIN
            else:
                user_create.role = UserRole.BASIC
        user = None
        try:
            user = await super().create(user_create, safe=safe, request=request)  # type: ignore
        except exceptions.UserAlreadyExists:
            user = await self.get_by_email(user_create.email)
            # Handle case where user has used product outside of web and is now creating an account through web
            if (
                not user.has_web_login
                and hasattr(user_create, "has_web_login")
                and user_create.has_web_login
            ):
                user_update = UserUpdate(
                    password=user_create.password,
                    has_web_login=True,
                    role=user_create.role,
                    is_verified=user_create.is_verified,
                )
                user = await self.update(user_update, user)
            else:
                raise exceptions.UserAlreadyExists()
        return user

    async def on_after_login(
        self,
        user: User,
        request: Request | None = None,
        response: Response | None = None,
    ) -> None:
        if response is None or not MULTI_TENANT:
            return

        tenant_id = get_tenant_id_for_email(user.email)

        tenant_token = jwt.encode(
            {"tenant_id": tenant_id}, SECRET_JWT_KEY, algorithm="HS256"
        )

        response.set_cookie(
            key="tenant_details",
            value=tenant_token,
            httponly=True,
            secure=WEB_DOMAIN.startswith("https"),
            samesite="lax",
        )

    async def oauth_callback(
        self: "BaseUserManager[models.UOAP, models.ID]",
        oauth_name: str,
        access_token: str,
        account_id: str,
        account_email: str,
        expires_at: Optional[int] = None,
        refresh_token: Optional[str] = None,
        request: Optional[Request] = None,
        *,
        associate_by_email: bool = False,
        is_verified_by_default: bool = False,
    ) -> models.UOAP:
        # Get tenant_id from mapping table
        try:
            tenant_id = (
                get_tenant_id_for_email(account_email) if MULTI_TENANT else "public"
            )
        except exceptions.UserNotExists:
            raise HTTPException(status_code=401, detail="User not found")

        if not tenant_id:
            raise HTTPException(status_code=401, detail="User not found")

        token = None
        async with get_async_session_with_tenant(tenant_id) as db_session:
            token = current_tenant_id.set(tenant_id)
            # Print a list of tables in the current database session
            verify_email_in_whitelist(account_email, tenant_id)
            verify_email_domain(account_email)
            if MULTI_TENANT:
                tenant_user_db = SQLAlchemyUserAdminDB(db_session, User, OAuthAccount)
                self.user_db = tenant_user_db
                self.database = tenant_user_db

            logger.info(f"Starting OAuth callback process for email: {account_email}")
            oauth_account_dict = {
                "oauth_name": oauth_name,
                "access_token": access_token,
                "account_id": account_id,
                "account_email": account_email,
                "expires_at": expires_at,
                "refresh_token": refresh_token,
            }
            logger.debug(f"OAuth account dict created: {oauth_account_dict}")

            try:
                logger.info(
                    f"Attempting to get user by OAuth account: {oauth_name}, {account_id}"
                )
                user = await self.get_by_oauth_account(oauth_name, account_id)
                logger.info(f"User found by OAuth account: {user.id}")
            except exceptions.UserNotExists:
                logger.info(
                    f"User not found by OAuth account, attempting to get by email: {account_email}"
                )
                try:
                    user = await self.get_by_email(account_email)
                    logger.info(f"User found by email: {user.id}")
                    if not associate_by_email:
                        logger.warning(
                            f"User already exists but associate_by_email is False: {account_email}"
                        )
                        raise exceptions.UserAlreadyExists()
                    logger.info(f"Adding OAuth account to existing user: {user.id}")
                    user = await self.user_db.add_oauth_account(
                        user, oauth_account_dict
                    )
                    logger.info(f"OAuth account added to user: {user.id}")
                except exceptions.UserNotExists:
                    logger.info(
                        f"User not found, creating new account for: {account_email}"
                    )
                    password = self.password_helper.generate()
                    user_dict = {
                        "email": account_email,
                        "hashed_password": self.password_helper.hash(password),
                        "is_verified": is_verified_by_default,
                    }
                    logger.debug(f"Creating new user with dict: {user_dict}")
                    user = await self.user_db.create(user_dict)
                    logger.info(f"New user created: {user.id}")
                    user = await self.user_db.add_oauth_account(
                        user, oauth_account_dict
                    )
                    logger.info(f"OAuth account added to new user: {user.id}")
                    await self.on_after_register(user, request)
            else:
                logger.info(f"Updating OAuth account for existing user: {user.id}")
                for existing_oauth_account in user.oauth_accounts:
                    if (
                        existing_oauth_account.account_id == account_id
                        and existing_oauth_account.oauth_name == oauth_name
                    ):
                        logger.info(
                            f"Updating OAuth account: {oauth_name}, {account_id}"
                        )
                        user = await self.user_db.update_oauth_account(
                            user, existing_oauth_account, oauth_account_dict
                        )
                        logger.info(f"OAuth account updated for user: {user.id}")

            logger.info("OAuth callback completed")

            try:
                if not user.has_web_login:  # type: ignore
                    await self.user_db.update(
                        user,
                        {
                            "is_verified": is_verified_by_default,
                            "has_web_login": True,
                        },
                    )
                    user.is_verified = is_verified_by_default
                    user.has_web_login = True  # type: ignore

                if expires_at and TRACK_EXTERNAL_IDP_EXPIRY:
                    oidc_expiry = datetime.fromtimestamp(expires_at, tz=timezone.utc)
                    await self.user_db.update(
                        user, update_dict={"oidc_expiry": oidc_expiry}
                    )

                if (
                    hasattr(user, "oidc_expiry")
                    and user.oidc_expiry is not None
                    and not TRACK_EXTERNAL_IDP_EXPIRY
                ):
                    await self.user_db.update(user, {"oidc_expiry": None})
                    user.oidc_expiry = None

            except Exception as e:
                logger.exception(f"Error in oauth_callback: {str(e)}")
            finally:
                if token:
                    current_tenant_id.reset(token)
            logger.info("oauth callback compleed")
            return user

    async def on_after_register(
        self, user: User, request: Optional[Request] = None
    ) -> None:
        logger.notice(f"User {user.id} has registered.")
        optional_telemetry(
            record_type=RecordType.SIGN_UP,
            data={"action": "create"},
            user_id=str(user.id),
        )

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ) -> None:
        logger.notice(f"User {user.id} has forgot their password. Reset token: {token}")

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ) -> None:
        verify_email_domain(user.email)

        logger.notice(
            f"Verification requested for user {user.id}. Verification token: {token}"
        )

        send_user_verification_email(user.email, token)

    async def authenticate(
        self, credentials: OAuth2PasswordRequestForm
    ) -> Optional[User]:
        email = credentials.username

        # Get tenant_id from mapping table
        tenant_id = get_tenant_id_for_email(email)
        if not tenant_id:
            # User not found in mapping
            self.password_helper.hash(credentials.password)
            return None

        # Create a tenant-specific session
        async with get_async_session_with_tenant(tenant_id) as tenant_session:
            tenant_user_db: SQLAlchemyUserDatabase = SQLAlchemyUserDatabase(
                tenant_session, User
            )
            self.user_db = tenant_user_db

            # Proceed with authentication
            try:
                user = await self.get_by_email(email)

            except exceptions.UserNotExists:
                self.password_helper.hash(credentials.password)
                return None

            has_web_login = attributes.get_attribute(user, "has_web_login")

            if not has_web_login:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="NO_WEB_LOGIN_AND_HAS_NO_PASSWORD",
                )

            verified, updated_password_hash = self.password_helper.verify_and_update(
                credentials.password, user.hashed_password
            )
            if not verified:
                return None

            if updated_password_hash is not None:
                await self.user_db.update(
                    user, {"hashed_password": updated_password_hash}
                )

            return user


async def get_user_manager(
    user_db: SQLAlchemyUserDatabase = Depends(get_user_db),
) -> AsyncGenerator[UserManager, None]:
    yield UserManager(user_db)


cookie_transport = CookieTransport(
    cookie_max_age=SESSION_EXPIRE_TIME_SECONDS,
    cookie_secure=WEB_DOMAIN.startswith("https"),
)


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(
        secret=USER_AUTH_SECRET,
        lifetime_seconds=SESSION_EXPIRE_TIME_SECONDS,
    )


def get_database_strategy(
    access_token_db: AccessTokenDatabase[AccessToken] = Depends(get_access_token_db),
) -> DatabaseStrategy:
    return DatabaseStrategy(
        access_token_db, lifetime_seconds=SESSION_EXPIRE_TIME_SECONDS  # type: ignore
    )


auth_backend = AuthenticationBackend(
    name="jwt" if MULTI_TENANT else "database",
    transport=cookie_transport,
    get_strategy=get_database_strategy if not MULTI_TENANT else get_jwt_strategy,  # type: ignore
)  # type: ignore


class FastAPIUserWithLogoutRouter(FastAPIUsers[models.UP, models.ID]):
    def get_logout_router(
        self,
        backend: AuthenticationBackend,
        requires_verification: bool = REQUIRE_EMAIL_VERIFICATION,
    ) -> APIRouter:
        """
        Provide a router for logout only for OAuth/OIDC Flows.
        This way the login router does not need to be included
        """
        router = APIRouter()

        get_current_user_token = self.authenticator.current_user_token(
            active=True, verified=requires_verification
        )

        logout_responses: OpenAPIResponseType = {
            **{
                status.HTTP_401_UNAUTHORIZED: {
                    "description": "Missing token or inactive user."
                }
            },
            **backend.transport.get_openapi_logout_responses_success(),
        }

        @router.post(
            "/logout", name=f"auth:{backend.name}.logout", responses=logout_responses
        )
        async def logout(
            user_token: Tuple[models.UP, str] = Depends(get_current_user_token),
            strategy: Strategy[models.UP, models.ID] = Depends(backend.get_strategy),
        ) -> Response:
            user, token = user_token
            return await backend.logout(strategy, user, token)

        return router


fastapi_users = FastAPIUserWithLogoutRouter[User, uuid.UUID](
    get_user_manager, [auth_backend]
)


# NOTE: verified=REQUIRE_EMAIL_VERIFICATION is not used here since we
# take care of that in `double_check_user` ourself. This is needed, since
# we want the /me endpoint to still return a user even if they are not
# yet verified, so that the frontend knows they exist
optional_fastapi_current_user = fastapi_users.current_user(active=True, optional=True)


async def optional_user_(
    request: Request,
    user: User | None,
    db_session: Session,
) -> User | None:
    """NOTE: `request` and `db_session` are not used here, but are included
    for the EE version of this function."""
    return user


async def optional_user(
    request: Request,
    db_session: Session = Depends(get_session),
    user: User | None = Depends(optional_fastapi_current_user),
) -> User | None:
    versioned_fetch_user = fetch_versioned_implementation(
        "danswer.auth.users", "optional_user_"
    )
    return await versioned_fetch_user(request, user, db_session)


async def double_check_user(
    user: User | None,
    optional: bool = DISABLE_AUTH,
    include_expired: bool = False,
) -> User | None:
    if optional:
        return None

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. User is not authenticated.",
        )

    if user_needs_to_be_verified() and not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. User is not verified.",
        )

    if (
        user.oidc_expiry
        and user.oidc_expiry < datetime.now(timezone.utc)
        and not include_expired
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. User's OIDC token has expired.",
        )

    return user


async def current_user_with_expired_token(
    user: User | None = Depends(optional_user),
) -> User | None:
    return await double_check_user(user, include_expired=True)


async def current_user(
    user: User | None = Depends(optional_user),
) -> User | None:
    return await double_check_user(user)


async def current_curator_or_admin_user(
    user: User | None = Depends(current_user),
) -> User | None:
    if DISABLE_AUTH:
        return None

    if not user or not hasattr(user, "role"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. User is not authenticated or lacks role information.",
        )

    allowed_roles = {UserRole.GLOBAL_CURATOR, UserRole.CURATOR, UserRole.ADMIN}
    if user.role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. User is not a curator or admin.",
        )

    return user


async def current_admin_user(user: User | None = Depends(current_user)) -> User | None:
    if DISABLE_AUTH:
        return None

    if not user or not hasattr(user, "role") or user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. User must be an admin to perform this action.",
        )

    return user


def get_default_admin_user_emails_() -> list[str]:
    # No default seeding available for Danswer MIT
    return []


async def control_plane_dep(request: Request) -> None:
    api_key = request.headers.get("X-API-KEY")
    if api_key != EXPECTED_API_KEY:
        logger.warning("Invalid API key")
        raise HTTPException(status_code=401, detail="Invalid API key")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.warning("Invalid authorization header")
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, DATA_PLANE_SECRET, algorithms=["HS256"])
        if payload.get("scope") != "tenant:create":
            logger.warning("Insufficient permissions")
            raise HTTPException(status_code=403, detail="Insufficient permissions")
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        logger.warning("Invalid token")
        raise HTTPException(status_code=401, detail="Invalid token")
