"""Standalone Method-2 form adapter, adapted from MIT-licensed Studio.

Only deterministic network/form parsing is retained. No Studio imports,
project model, provider, or LLM is required. See ../STUDIO-LICENSE.
"""
from __future__ import annotations
import hashlib,html,io,itertools,json,math,re,time
from fractions import Fraction
from html.parser import HTMLParser
from typing import Any,Callable,Literal
import requests


ISODISTORT_UPLOAD_URL = 'https://iso.byu.edu/isodistortuploadfile.php'
ISODISTORT_FORM_URL = 'https://iso.byu.edu/isodistortform.php'
ISOCIF_UPLOAD_URL = 'https://iso.byu.edu/isocifuploadfile.php'
ISOCIF_FORM_URL = 'https://iso.byu.edu/isocifform.php'
_MAX_RESPONSE_BYTES = 20 * 1024 * 1024

def _fraction_text(value: str, *, label: str, maximum: float=128.0) -> str:
    normalized = str(value).strip()
    if not normalized or not re.fullmatch('[0-9+\\-./]+', normalized):
        raise ValueError(f'{label} must be an integer, decimal, or rational number')
    try:
        number = float(Fraction(normalized)) if '/' in normalized else float(normalized)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f'{label} is not a valid number') from exc
    if not math.isfinite(number) or abs(number) > maximum:
        raise ValueError(f'{label} must be finite and have absolute value <= {maximum:g}')
    return normalized

class ExternalServiceError(RuntimeError):
    """A sanitized failure from a fixed external crystallographic service."""

    def __init__(self, message: str, *, raw_responses: dict[str, str] | None=None, diagnostics: dict[str, Any] | None=None) -> None:
        super().__init__(message)
        self.raw_responses = dict(raw_responses or {})
        self.diagnostics = dict(diagnostics or {})

class _InputCollector(HTMLParser):

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.inputs: list[dict[str, str]] = []
        self.options: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        mapped = {str(key).casefold(): '' if value is None else str(value) for (key, value) in attrs}
        if tag.casefold() == 'input':
            self.inputs.append(mapped)
        elif tag.casefold() == 'option':
            self.options.append(mapped)

def _collect_inputs(markup: str) -> _InputCollector:
    parser = _InputCollector()
    parser.feed(markup)
    return parser

def _upload_filename(markup: str) -> str:
    inputs = _collect_inputs(markup).inputs
    for item in inputs:
        value = item.get('value', '').strip()
        if value:
            return value
    raise ExternalServiceError('ISODISTORT upload response lacks a temporary filename')

def _initial_form_fields(markup: str) -> dict[str, str]:
    folded = markup.casefold()
    if '<form' not in folded:
        raise ExternalServiceError('ISODISTORT parent response lacks a form')
    fields: dict[str, str] = {}
    for item in _collect_inputs(markup).inputs:
        name = item.get('name', '').strip()
        if not name:
            continue
        if 'isosystem' in name.casefold():
            break
        input_type = item.get('type', '').casefold()
        if input_type == 'checkbox' and 'checked' not in item:
            continue
        if input_type in {'submit', 'button', 'radio', 'file', 'image'}:
            continue
        fields[name] = item.get('value', '')
    if not fields:
        raise ExternalServiceError('ISODISTORT parent form contains no reusable fields')
    return fields

def _strip_markup(value: str) -> str:
    return ' '.join(html.unescape(re.sub('<[^>]+>', ' ', value)).split())

def _method1_catalog_group(label: str) -> dict[str, Any]:
    """Extract the child group printed by an authoritative Method-1 row."""
    match = re.search('\\)\\s+(?P<number>\\d{1,3})\\s+(?P<symbol>[^,]+?)\\s*,\\s*basis\\s*=', label, flags=re.IGNORECASE)
    if match is None:
        return {}
    number = int(match.group('number'))
    if not 1 <= number <= 230:
        return {}
    symbol = ' '.join(match.group('symbol').split())
    if not symbol:
        return {}
    return {'catalog_child_space_group_number': number, 'catalog_child_space_group_symbol': symbol, 'catalog_group_authority': 'isodistort-method1-row'}

def _method2_catalog_metadata(label: str) -> dict[str, Any]:
    """Parse and validate the transformation printed by a Method-2 OPD row."""
    group = _method1_catalog_group(label)
    basis_match = re.search('\\bbasis\\s*=\\s*\\{(?P<basis>[^{}]+)\\}\\s*,\\s*origin\\s*=\\s*\\((?P<origin>[^()]*)\\)\\s*,\\s*s\\s*=\\s*(?P<s>\\d+)\\s*,\\s*i\\s*=\\s*(?P<i>\\d+)', label, flags=re.IGNORECASE)
    if not group or basis_match is None:
        return {}
    k_active_match = re.search('\\bk-active\\s*=\\s*(?P<vectors>.*)$', label, flags=re.IGNORECASE)
    row_texts = re.findall('\\(([^()]*)\\)', basis_match.group('basis'))
    if len(row_texts) != 3:
        return {}
    try:
        basis_fraction = [[Fraction(value.strip()) for value in row.split(',')] for row in row_texts]
        origin_fraction = [Fraction(value.strip()) for value in basis_match.group('origin').split(',')]
        k_active_fraction = [[Fraction(value.strip()) for value in row.split(',')] for row in re.findall('\\(([^()]*)\\)', k_active_match.group('vectors'))] if k_active_match is not None else []
    except (ValueError, ZeroDivisionError):
        return {}
    if any((len(row) != 3 for row in basis_fraction)) or len(origin_fraction) != 3:
        return {}
    if any((len(row) != 3 for row in k_active_fraction)):
        return {}
    determinant = basis_fraction[0][0] * (basis_fraction[1][1] * basis_fraction[2][2] - basis_fraction[1][2] * basis_fraction[2][1]) - basis_fraction[0][1] * (basis_fraction[1][0] * basis_fraction[2][2] - basis_fraction[1][2] * basis_fraction[2][0]) + basis_fraction[0][2] * (basis_fraction[1][0] * basis_fraction[2][1] - basis_fraction[1][1] * basis_fraction[2][0])
    lattice_index = int(basis_match.group('s'))
    subgroup_index = int(basis_match.group('i'))
    if determinant == 0:
        raise ExternalServiceError('ISODISTORT catalog row has a singular basis transformation')
    if subgroup_index < lattice_index or subgroup_index % lattice_index:
        raise ExternalServiceError('ISODISTORT catalog row has inconsistent lattice and subgroup indices')
    return {**group, 'catalog_group_authority': 'isodistort-method2-row', 'catalog_basis': [[float(value) for value in row] for row in basis_fraction], 'catalog_basis_rational': [[str(value) for value in row] for row in basis_fraction], 'catalog_origin': [float(value) for value in origin_fraction], 'catalog_origin_rational': [str(value) for value in origin_fraction], 'catalog_lattice_index': lattice_index, 'catalog_subgroup_index': subgroup_index, 'catalog_k_active_vectors': [[float(value) for value in row] for row in k_active_fraction], 'catalog_k_active_vectors_rational': [[str(value) for value in row] for row in k_active_fraction]}
_METHOD2_CATALOG_KEYS = ('catalog_child_space_group_number', 'catalog_child_space_group_symbol', 'catalog_group_authority', 'catalog_basis', 'catalog_basis_rational', 'catalog_origin', 'catalog_origin_rational', 'catalog_lattice_index', 'catalog_subgroup_index', 'catalog_k_active_vectors', 'catalog_k_active_vectors_rational')

def _method2_k_activity_is_consistent(requested_k_vector: tuple[float, float, float], catalog_k_active_vectors: list[list[float]]) -> bool:
    """Require returned Method-2 rows to preserve Gamma/non-Gamma character."""

    def is_gamma(vector: list[float] | tuple[float, float, float]) -> bool:
        return all((abs(float(value) - round(float(value))) <= 1e-08 for value in vector))
    if not catalog_k_active_vectors:
        return True
    requested_is_gamma = is_gamma(requested_k_vector)
    returned_is_gamma = all((is_gamma(vector) for vector in catalog_k_active_vectors))
    return requested_is_gamma == returned_is_gamma

def _named_input_value(markup: str, name: str, *, allow_empty: bool=False) -> str:
    for item in _collect_inputs(markup).inputs:
        if item.get('name', '').strip().casefold() == name.casefold():
            value = item.get('value', '').strip()
            if value or allow_empty:
                return value
    raise ExternalServiceError(f'ISODISTORT response lacks input {name!r}')

def _select_options(markup: str, name: str) -> list[dict[str, str]]:
    pattern = re.compile(f"""<select\\b[^>]*\\bname\\s*=\\s*[\\"']?{re.escape(name)}[\\"']?[^>]*>(?P<body>.*?)</select\\s*>""", flags=re.IGNORECASE | re.DOTALL)
    match = pattern.search(markup)
    if match is None:
        raise ExternalServiceError(f'ISODISTORT response lacks select {name!r}')
    option_pattern = re.compile('<option\\b(?P<attrs>[^>]*)>(?P<label>.*?)(?=<option\\b|</select\\s*>|$)', flags=re.IGNORECASE | re.DOTALL)
    result = []
    for option in option_pattern.finditer(match.group('body')):
        parsed = _collect_inputs('<option ' + option.group('attrs') + '>').options
        if not parsed:
            continue
        value = parsed[0].get('value', '').strip()
        if value:
            result.append({'value': value, 'label': _strip_markup(option.group('label')) or value})
    if not result:
        raise ExternalServiceError(f'ISODISTORT select {name!r} contains no options')
    return result

def _method2_specialized_k_option(markup: str, options: list[dict[str, str]]) -> str:
    """Read ISODISTORT's authoritative special-k redirect, if present.

    ISODISTORT rejects a general k-form when the submitted parameters land on a
    named special point (for example ``A(a,b,0)`` at ``(1/2,0,0)`` is ``X``).
    The service reports the correct label in a bounded diagnostic. Resolve that
    label only against the options returned by the same parent session.
    """
    message = _strip_markup(markup)
    match = re.search('puts\\s+the\\s+k\\s+point\\s+at\\s+([A-Za-z][A-Za-z0-9_+\\-]*)\\s+not\\s+([A-Za-z][A-Za-z0-9_+\\-]*)', message, flags=re.IGNORECASE)
    if match is None:
        return ''
    recommended = match.group(1).casefold()
    matches = []
    for option in options:
        value = str(option.get('value') or '')
        label = str(option.get('label') or '')
        name_match = re.search('\\*\\s*([A-Za-z][A-Za-z0-9_+\\-]*)\\s*,', value)
        label_match = re.match('\\s*([A-Za-z][A-Za-z0-9_+\\-]*)\\s*,', label)
        option_name = name_match.group(1) if name_match is not None else label_match.group(1) if label_match is not None else ''
        if option_name.casefold() == recommended:
            matches.append(value)
    if len(matches) != 1:
        raise ExternalServiceError('ISODISTORT recommended a special k form that is absent or ambiguous in the current parent-session options')
    return matches[0]

def _radio_values(markup: str, name: str) -> list[str]:
    result = [item.get('value', '').strip() for item in _collect_inputs(markup).inputs if item.get('type', '').casefold() == 'radio' and item.get('name', '').strip().casefold() == name.casefold() and item.get('value', '').strip()]
    if not result:
        raise ExternalServiceError(f'ISODISTORT response lacks radio values for {name!r}')
    return result

def _form_fields_containing(markup: str, marker: str, *, purpose: str) -> dict[str, str]:
    forms = re.findall('<form\\b[^>]*>.*?</form\\s*>', markup, flags=re.IGNORECASE | re.DOTALL)
    form = next((value for value in forms if marker.casefold() in value.casefold()), '')
    if not form:
        raise ExternalServiceError(f'{purpose} response lacks the expected form')
    fields: dict[str, str] = {}
    for item in _collect_inputs(form).inputs:
        name = item.get('name', '').strip()
        if not name:
            continue
        input_type = item.get('type', '').casefold()
        if input_type in {'submit', 'button', 'radio', 'file', 'image'}:
            continue
        if input_type == 'checkbox' and 'checked' not in item:
            continue
        fields[name] = item.get('value', '')
    if not fields:
        raise ExternalServiceError(f'{purpose} form contains no reusable fields')
    return fields

def _cif_request_fields(markup: str, *, method: Literal[1, 4]) -> dict[str, Any]:
    form_position = markup.casefold().find('<form')
    if form_position < 0:
        raise ExternalServiceError('ISODISTORT distortion response lacks a CIF form')
    fields: dict[str, Any] = {}
    inputs = _collect_inputs(markup[form_position:]).inputs
    origin_index = -1
    for (index, item) in enumerate(inputs):
        name = item.get('name', '').strip()
        if not name:
            continue
        input_type = item.get('type', '').casefold()
        if input_type not in {'submit', 'button', 'image', 'file', 'radio'}:
            fields[name] = item.get('value', '')
        if 'subsetting' in name.casefold():
            fields[name] = ''
        if 'atomsfile' in name.casefold():
            fields[name] = ' '
        if 'origintype' in name.casefold():
            origin_index = index
            break
    if origin_index < 0:
        raise ExternalServiceError('ISODISTORT CIF form lacks origintype')
    if method == 4:
        for item in inputs[origin_index + 1:]:
            name = item.get('name', '').strip()
            if not name:
                continue
            if name == 'atomicradius':
                break
            input_type = item.get('type', '').casefold()
            if input_type not in {'submit', 'button', 'image', 'file', 'radio'}:
                fields[name] = item.get('value', '')
        fields['zeromodes'] = 'false'
    fields.update({'origintype': 'structurefile', 'inputvalues': 'false', 'atomicradius': '0.4', 'bondlength': '2.50', 'modeamplitude': '1.0', 'strainamplitude': '0.1'})
    fields.pop('appbtn', None)
    return fields

def _validate_isodistort_cif(value: str) -> str:
    text = str(value).strip() + '\n'
    folded = text.casefold()
    if not re.search('(?m)^\\s*data_', text, flags=re.IGNORECASE):
        raise ExternalServiceError('ISODISTORT final response is not a CIF data block')
    required = ('_cell_length_a', '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z')
    if not all((marker in folded for marker in required)):
        raise ExternalServiceError('ISODISTORT final CIF lacks required structure fields')
    if len(text.encode('utf-8')) > _MAX_RESPONSE_BYTES:
        raise ExternalServiceError('ISODISTORT CIF exceeded the 20 MiB safety limit')
    return text

class IsodistortFormClient:
    """Versioned adapter for the official multi-step ISODISTORT HTML workflow."""

    def __init__(self, *, post: Callable[..., Any]=requests.post, timeout_s: float=120.0, max_attempts: int=3, retry_backoff_s: float=0.5, sleep: Callable[[float], Any]=time.sleep) -> None:
        self._post_callable = post
        self.timeout_s = float(timeout_s)
        self.max_attempts = max(1, min(4, int(max_attempts)))
        self.retry_backoff_s = max(0.0, min(5.0, float(retry_backoff_s)))
        self._sleep = sleep
        self.request_trace: list[dict[str, Any]] = []
        self.raw_responses: dict[str, str] = {}

    @staticmethod
    def _retryable_request_error(exc: BaseException) -> tuple[bool, int | None]:
        response = getattr(exc, 'response', None)
        status_code = getattr(response, 'status_code', None)
        if isinstance(status_code, int):
            return (status_code in {408, 425, 429, 500, 502, 503, 504}, status_code)
        return (isinstance(exc, (requests.ConnectionError, requests.Timeout, OSError)), None)

    @staticmethod
    def _safe_error_detail(exc: BaseException) -> str:
        detail = ' '.join(str(exc).split())
        return detail[:500] or type(exc).__name__

    def _post(self, url: str, *, step: str, **kwargs: Any) -> str:
        endpoint_names = {ISODISTORT_UPLOAD_URL: 'isodistort_upload', ISODISTORT_FORM_URL: 'isodistort_form', ISOCIF_UPLOAD_URL: 'isocif_upload', ISOCIF_FORM_URL: 'isocif_form'}
        if url not in endpoint_names:
            raise ExternalServiceError('ISODISTORT/ISOCIF endpoint is outside the fixed allowlist')
        endpoint = endpoint_names[url]
        for attempt in range(1, self.max_attempts + 1):
            started = time.monotonic()
            try:
                response = self._post_callable(url, timeout=self.timeout_s, **kwargs)
                response.raise_for_status()
                raw = bytes(response.content)
            except (requests.RequestException, OSError) as exc:
                (retryable, status_code) = self._retryable_request_error(exc)
                detail = self._safe_error_detail(exc)
                trace_item = {'step': step, 'endpoint': endpoint, 'attempt': attempt, 'status': 'failed', 'retryable': retryable, 'error_type': type(exc).__name__, 'error_detail': detail, 'elapsed_seconds': round(max(0.0, time.monotonic() - started), 3)}
                if status_code is not None:
                    trace_item['http_status'] = status_code
                self.request_trace.append(trace_item)
                if retryable and attempt < self.max_attempts:
                    self._sleep(self.retry_backoff_s * 2 ** (attempt - 1))
                    continue
                diagnostics = {'service': 'ISODISTORT', 'failed_step': step, 'endpoint': endpoint, 'attempts': attempt, 'max_attempts': self.max_attempts, 'retryable': retryable, 'error_type': type(exc).__name__, 'error_detail': detail, 'http_status': status_code, 'request_trace': list(self.request_trace[-24:])}
                raise ExternalServiceError(f'ISODISTORT {step} failed after {attempt} attempt(s): {type(exc).__name__}: {detail}', raw_responses=self.raw_responses, diagnostics=diagnostics) from exc
            self.request_trace.append({'step': step, 'endpoint': endpoint, 'attempt': attempt, 'status': 'ok', 'response_bytes': len(raw), 'elapsed_seconds': round(max(0.0, time.monotonic() - started), 3)})
            break
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ExternalServiceError('ISODISTORT response exceeded the 20 MiB safety limit', raw_responses=self.raw_responses, diagnostics={'service': 'ISODISTORT', 'failed_step': step, 'endpoint': endpoint, 'attempts': attempt, 'max_attempts': self.max_attempts, 'retryable': False, 'error_type': 'ResponseTooLarge', 'request_trace': list(self.request_trace[-24:])})
        text = raw.decode(getattr(response, 'encoding', None) or 'utf-8', errors='replace')
        self.raw_responses[step] = text
        return text

    def upload_cif(self, filename: str, content: bytes, *, step: str='upload_cif', upload_url: str=ISODISTORT_UPLOAD_URL) -> tuple[str, str]:
        if not filename.lower().endswith('.cif'):
            raise ValueError('ISODISTORT accepts registered CIF artifacts only')
        if not content or len(content) > 5 * 1024 * 1024:
            raise ValueError('ISODISTORT CIF upload must be between 1 byte and 5 MiB')
        markup = self._post(upload_url, step=step, files={'toProcess': (filename, io.BytesIO(content), 'chemical/x-cif')})
        return (_upload_filename(markup), markup)

    def _parent_form(self, temporary_filename: str, *, step: str='parent_form') -> tuple[dict[str, str], str]:
        markup = self._post(ISODISTORT_FORM_URL, step=step, data={'filename': temporary_filename, 'input': 'uploadparentcif'})
        normalized = _strip_markup(markup).casefold()
        remote_cif_error = next((marker for marker in ('error reading symmetry operators', 'error reading cif', 'program has bombed') if marker in normalized), '')
        if remote_cif_error:
            attempts = sum((1 for item in self.request_trace if str(item.get('step') or '') == step)) or 1
            raise ExternalServiceError('ISODISTORT rejected the CIF while reading symmetry operators', raw_responses=self.raw_responses, diagnostics={'service': 'ISODISTORT', 'failed_step': step, 'endpoint': 'form', 'attempts': attempts, 'max_attempts': self.max_attempts, 'retryable': False, 'error_type': 'RemoteCifParseError', 'error_detail': remote_cif_error, 'request_trace': list(self.request_trace[-24:])})
        return (_initial_form_fields(markup), markup)

    def method2_k_vector_search(self, filename: str, content: bytes, *, k_vector: tuple[float, float, float], k_form_matcher: Callable[[str, tuple[float, float, float], list[str]], dict[str, str]]) -> dict[str, Any]:
        """Enumerate exact ISODISTORT irreps/OPDs for one native k-vector."""
        if len(k_vector) != 3 or any((not math.isfinite(float(value)) for value in k_vector)):
            raise ValueError('k_vector must contain three finite components')
        (temporary, upload_html) = self.upload_cif(filename, content, step='method2_parent_upload')
        (base_fields, parent_html) = self._parent_form(temporary, step='method2_parent_form')
        (isocif_temporary, isocif_upload_html) = self.upload_cif(filename, content, step='method2_isocif_upload', upload_url=ISOCIF_UPLOAD_URL)
        isocif_entry_html = self._post(ISOCIF_FORM_URL, step='method2_isocif_entry', data={'submit': 'OK', 'input': 'uploadcif', 'filename': isocif_temporary})
        isocif_fields = _form_fields_containing(isocif_entry_html, 'Detect higher', purpose='ISOCIF standardization')
        isocif_fields['input'] = 'savecif'
        standardized_cif_response = self._post(ISOCIF_FORM_URL, step='method2_isocif_standardized_cif', data=isocif_fields)
        standardized_cif = _validate_isodistort_cif(standardized_cif_response)
        k_options = _select_options(parent_html, 'kvec1')
        option_values = [item['value'] for item in k_options]
        matched = k_form_matcher(standardized_cif, tuple((float(value) for value in k_vector)), option_values)
        if not isinstance(matched, dict) or not str(matched.get('kvec1') or ''):
            raise ExternalServiceError('native GSAS-II k-form matcher returned no kvec1')
        if str(matched['kvec1']) not in option_values:
            raise ExternalServiceError('native GSAS-II k-form matcher selected an unknown form')
        permitted_match_fields = {'kvec1', 'kparam11', 'kparam21', 'kparam31'}
        if set(matched).difference(permitted_match_fields):
            raise ExternalServiceError('native GSAS-II k-form matcher returned unsupported fields')
        for (key, value) in matched.items():
            if key != 'kvec1':
                _fraction_text(str(value), label=key, maximum=1024.0)
        data = {str(key): str(value) for (key, value) in base_fields.items()}
        data.update({'input': 'kvector', 'irrepcount': '1', 'nmodstar1': '0', **{str(key): str(value) for (key, value) in matched.items()}})
        irrep_selection_html = self._post(ISODISTORT_FORM_URL, step='method2_irrep_selection', data=data)
        specialization_html = ''
        specialized_option = _method2_specialized_k_option(irrep_selection_html, k_options)
        if specialized_option and specialized_option != data.get('kvec1'):
            expression = specialized_option.split('(', 1)[-1].rsplit(')', 1)[0]
            if re.search('\\b[abg]\\b', expression):
                specialized_fields = k_form_matcher(standardized_cif, tuple((float(value) for value in k_vector)), [specialized_option])
                if str(specialized_fields.get('kvec1') or '') != specialized_option:
                    raise ExternalServiceError('native GSAS-II matcher could not parameterize the special k form required by ISODISTORT')
            else:
                specialized_fields = {'kvec1': specialized_option}
            for key in ('kvec1', 'kparam11', 'kparam21', 'kparam31', 'kparam1'):
                data.pop(key, None)
            data.update({str(key): str(value) for (key, value) in specialized_fields.items()})
            specialization_html = self._post(ISODISTORT_FORM_URL, step='method2_k_specialization', data=data)
            if _method2_specialized_k_option(specialization_html, k_options):
                raise ExternalServiceError('ISODISTORT rejected its own specialized k-form redirect')
            irrep_selection_html = specialization_html
        for name in ('kvec1', 'kvecnumber1', 'kparam1', 'nmodstar1'):
            data[name] = _named_input_value(irrep_selection_html, name, allow_empty=name == 'kparam1')
        irrep_options = _select_options(irrep_selection_html, 'irrep1')
        if len(irrep_options) > 512:
            raise ExternalServiceError('ISODISTORT returned too many irreps')
        candidates: list[dict[str, Any]] = []
        irrep_failures: list[dict[str, Any]] = []
        raw_responses: dict[str, str] = {'parent_upload': upload_html, 'parent_form': parent_html, 'isocif_upload': isocif_upload_html, 'isocif_entry': isocif_entry_html, 'isocif_standardized_cif': standardized_cif_response, 'irrep_selection': irrep_selection_html}
        if specialization_html:
            raw_responses['k_specialization'] = specialization_html
        for (irrep_index, irrep) in enumerate(irrep_options, 1):
            irrep_data = dict(data)
            irrep_data.update({'input': 'irrep', 'irrep1': irrep['value']})
            distortion_html = self._post(ISODISTORT_FORM_URL, step=f'method2_irrep_{irrep_index}', data=irrep_data)
            raw_responses[f'irrep_{irrep_index}'] = distortion_html
            try:
                irrep_fields = {'irrep1': _named_input_value(distortion_html, 'irrep1'), 'irrpointer1': _named_input_value(distortion_html, 'irrpointer1'), 'isofilename': _named_input_value(distortion_html, 'isofilename', allow_empty=True)}
                order_parameters = _radio_values(distortion_html, 'orderparam')
                if not order_parameters:
                    raise ExternalServiceError('ISODISTORT irrep response contains no order parameters')
            except ExternalServiceError as exc:
                irrep_failures.append({'irrep_index': irrep_index, 'irrep': irrep['value'], 'irrep_label': irrep['label'], 'failure_type': type(exc).__name__, 'summary': str(exc), 'retryable': False})
                continue
            for order_parameter in order_parameters:
                candidate_id = 'iso2:' + hashlib.sha256((irrep['value'] + '\n' + order_parameter).encode('utf-8')).hexdigest()[:16]
                candidates.append({'candidate_id': candidate_id, 'label': f"{irrep['label']} · {_strip_markup(order_parameter)}", 'irrep': irrep['value'], 'irrep_label': irrep['label'], 'order_parameter': order_parameter, 'irrep_fields': irrep_fields})
                candidates[-1].update(_method2_catalog_metadata(candidates[-1]['label']))
        activity_compatible = []
        for candidate in candidates:
            active_vectors = candidate.get('catalog_k_active_vectors')
            if active_vectors and (not _method2_k_activity_is_consistent(k_vector, active_vectors)):
                irrep_failures.append({'irrep': candidate['irrep'], 'irrep_label': candidate['irrep_label'], 'failure_type': 'InconsistentKActiveVector', 'summary': 'ISODISTORT returned a Gamma/non-Gamma k-active row inconsistent with the requested propagation vector', 'requested_k_vector': [float(value) for value in k_vector], 'catalog_k_active_vectors': active_vectors, 'retryable': False})
                continue
            activity_compatible.append(candidate)
        candidates = activity_compatible
        if not candidates:
            raise ExternalServiceError('ISODISTORT Method-2 response contains no usable order-parameter candidates', raw_responses=raw_responses, diagnostics={'service': 'ISODISTORT', 'failed_step': 'method2_irrep_enumeration', 'retryable': False, 'error_type': 'UnsupportedKVectorCatalog', 'irrep_failures': irrep_failures[:64], 'request_trace': list(self.request_trace[-24:])})
        if len(candidates) > 10000:
            raise ExternalServiceError('ISODISTORT returned an implausible candidate count')
        session_candidates = [{'candidate_id': item['candidate_id'], 'label': item['label'], 'irrep': item['irrep'], 'irrep_label': item['irrep_label'], 'order_parameter': item['order_parameter'], 'irrep_fields': item['irrep_fields'], **{key: item[key] for key in _METHOD2_CATALOG_KEYS if key in item}} for item in candidates]
        return {'session_state': {'common_fields': data, 'candidates': session_candidates, 'k_vector': [float(value) for value in k_vector], 'k_form_fields': {str(key): str(value) for (key, value) in matched.items()}}, 'candidates': [{key: item[key] for key in ('candidate_id', 'label', 'irrep', 'irrep_label', 'order_parameter') + _METHOD2_CATALOG_KEYS if key in item} for item in candidates], 'standardized_parent_cif': standardized_cif, 'irrep_failures': irrep_failures, 'raw_responses': raw_responses}

    def method2_generate(self, session_state: dict[str, Any], candidate_id: str) -> dict[str, Any]:
        common_fields = session_state.get('common_fields')
        candidates = session_state.get('candidates')
        if not isinstance(common_fields, dict) or not isinstance(candidates, list):
            raise ValueError('invalid persisted ISODISTORT Method-2 session')
        selected = next((item for item in candidates if isinstance(item, dict) and item.get('candidate_id') == candidate_id), None)
        if selected is None:
            raise ValueError('candidate_id is absent from the persisted Method-2 session')
        data = {str(key): str(value) for (key, value) in common_fields.items()}
        irrep_fields = selected.get('irrep_fields')
        if not isinstance(irrep_fields, dict):
            raise ValueError('persisted Method-2 candidate lacks irrep fields')
        data.update({str(key): str(value) for (key, value) in irrep_fields.items()})
        data.update({'input': 'distort', 'origintype': 'method2', 'orderparam': str(selected['order_parameter']) + '" CHECKED'})
        distortion_html = self._post(ISODISTORT_FORM_URL, step='method2_generate_distortion', data=data)
        cif_fields = _cif_request_fields(distortion_html, method=1)
        cif_response = self._post(ISODISTORT_FORM_URL, step='method2_generate_cif', data=cif_fields)
        return {'candidate': {key: selected.get(key, '') for key in ('candidate_id', 'label', 'irrep', 'irrep_label', 'order_parameter')}, 'cif': _validate_isodistort_cif(cif_response), 'raw_responses': {'distortion': distortion_html, 'cif_response': cif_response}}
