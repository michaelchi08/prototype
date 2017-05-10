# prototype.tests.models.two_wheel_test

## Classes

- TwoWheelTest


### TwoWheelTest


    __call__(self, *args, **kwds)


---

    __eq__(self, other)


---

    __hash__(self)


---

    __init__(self, methodName='runTest')

Create an instance of the class that will use the named test
           method when executed. Raises a ValueError if the instance does
           not have a method with the specified name.
        

---

    __repr__(self)


---

    __str__(self)


---

    _addExpectedFailure(self, result, exc_info)


---

    _addSkip(self, result, test_case, reason)


---

    _addUnexpectedSuccess(self, result)


---

    _baseAssertEqual(self, first, second, msg=None)

The default assertEqual implementation, not type specific.

---

    _deprecate(original_func)


---

    _feedErrorsToResult(self, result, errors)


---

    _formatMessage(self, msg, standardMsg)

Honour the longMessage attribute when generating failure messages.
        If longMessage is False this means:
        * Use only an explicit message if it is provided
        * Otherwise use the standard message for the assert

        If longMessage is True:
        * Use the standard message
        * If an explicit message is provided, plus ' : ' and the explicit message
        

---

    _getAssertEqualityFunc(self, first, second)

Get a detailed comparison function for the types of the two args.

        Returns: A callable accepting (first, second, msg=None) that will
        raise a failure exception if first != second with a useful human
        readable error message for those types.
        

---

    _truncateMessage(self, message, diff)


---

    addCleanup(self, function, *args, **kwargs)

Add a function, with arguments, to be called when the test is
        completed. Functions added are called on a LIFO basis and are
        called after tearDown on test failure or success.

        Cleanup items are called even if setUp fails (unlike tearDown).

---

    addTypeEqualityFunc(self, typeobj, function)

Add a type specific assertEqual style function to compare a type.

        This method is for use by TestCase subclasses that need to register
        their own type equality functions to provide nicer error messages.

        Args:
            typeobj: The data type to call this function on when both values
                    are of the same type in assertEqual().
            function: The callable taking two arguments and an optional
                    msg= argument that raises self.failureException with a
                    useful error message when the two arguments are not equal.
        

---

    assertAlmostEqual(self, first, second, places=None, msg=None, delta=None)

Fail if the two objects are unequal as determined by their
           difference rounded to the given number of decimal places
           (default 7) and comparing to zero, or by comparing that the
           between the two objects is more than the given delta.

           Note that decimal places (from zero) are usually not the same
           as significant digits (measured from the most signficant digit).

           If the two objects compare equal then they will automatically
           compare almost equal.
        

---

    assertAlmostEquals(*args, **kwargs)


---

    assertCountEqual(self, first, second, msg=None)

An unordered sequence comparison asserting that the same elements,
        regardless of order.  If the same element occurs more than once,
        it verifies that the elements occur the same number of times.

            self.assertEqual(Counter(list(first)),
                             Counter(list(second)))

         Example:
            - [0, 1, 1] and [1, 0, 1] compare equal.
            - [0, 0, 1] and [0, 1] compare unequal.

        

---

    assertDictContainsSubset(self, subset, dictionary, msg=None)

Checks whether dictionary is a superset of subset.

---

    assertDictEqual(self, d1, d2, msg=None)


---

    assertEqual(self, first, second, msg=None)

Fail if the two objects are unequal as determined by the '=='
           operator.
        

---

    assertEquals(*args, **kwargs)


---

    assertFalse(self, expr, msg=None)

Check that the expression is false.

---

    assertGreater(self, a, b, msg=None)

Just like self.assertTrue(a > b), but with a nicer default message.

---

    assertGreaterEqual(self, a, b, msg=None)

Just like self.assertTrue(a >= b), but with a nicer default message.

---

    assertIn(self, member, container, msg=None)

Just like self.assertTrue(a in b), but with a nicer default message.

---

    assertIs(self, expr1, expr2, msg=None)

Just like self.assertTrue(a is b), but with a nicer default message.

---

    assertIsInstance(self, obj, cls, msg=None)

Same as self.assertTrue(isinstance(obj, cls)), with a nicer
        default message.

---

    assertIsNone(self, obj, msg=None)

Same as self.assertTrue(obj is None), with a nicer default message.

---

    assertIsNot(self, expr1, expr2, msg=None)

Just like self.assertTrue(a is not b), but with a nicer default message.

---

    assertIsNotNone(self, obj, msg=None)

Included for symmetry with assertIsNone.

---

    assertLess(self, a, b, msg=None)

Just like self.assertTrue(a < b), but with a nicer default message.

---

    assertLessEqual(self, a, b, msg=None)

Just like self.assertTrue(a <= b), but with a nicer default message.

---

    assertListEqual(self, list1, list2, msg=None)

A list-specific equality assertion.

        Args:
            list1: The first list to compare.
            list2: The second list to compare.
            msg: Optional message to use on failure instead of a list of
                    differences.

        

---

    assertLogs(self, logger=None, level=None)

Fail unless a log message of level *level* or higher is emitted
        on *logger_name* or its children.  If omitted, *level* defaults to
        INFO and *logger* defaults to the root logger.

        This method must be used as a context manager, and will yield
        a recording object with two attributes: `output` and `records`.
        At the end of the context manager, the `output` attribute will
        be a list of the matching formatted log messages and the
        `records` attribute will be a list of the corresponding LogRecord
        objects.

        Example::

            with self.assertLogs('foo', level='INFO') as cm:
                logging.getLogger('foo').info('first message')
                logging.getLogger('foo.bar').error('second message')
            self.assertEqual(cm.output, ['INFO:foo:first message',
                                         'ERROR:foo.bar:second message'])
        

---

    assertMultiLineEqual(self, first, second, msg=None)

Assert that two multi-line strings are equal.

---

    assertNotAlmostEqual(self, first, second, places=None, msg=None, delta=None)

Fail if the two objects are equal as determined by their
           difference rounded to the given number of decimal places
           (default 7) and comparing to zero, or by comparing that the
           between the two objects is less than the given delta.

           Note that decimal places (from zero) are usually not the same
           as significant digits (measured from the most signficant digit).

           Objects that are equal automatically fail.
        

---

    assertNotAlmostEquals(*args, **kwargs)


---

    assertNotEqual(self, first, second, msg=None)

Fail if the two objects are equal as determined by the '!='
           operator.
        

---

    assertNotEquals(*args, **kwargs)


---

    assertNotIn(self, member, container, msg=None)

Just like self.assertTrue(a not in b), but with a nicer default message.

---

    assertNotIsInstance(self, obj, cls, msg=None)

Included for symmetry with assertIsInstance.

---

    assertNotRegex(self, text, unexpected_regex, msg=None)

Fail the test if the text matches the regular expression.

---

    assertNotRegexpMatches(*args, **kwargs)


---

    assertRaises(self, expected_exception, *args, **kwargs)

Fail unless an exception of class expected_exception is raised
           by the callable when invoked with specified positional and
           keyword arguments. If a different type of exception is
           raised, it will not be caught, and the test case will be
           deemed to have suffered an error, exactly as for an
           unexpected exception.

           If called with the callable and arguments omitted, will return a
           context object used like this::

                with self.assertRaises(SomeException):
                    do_something()

           An optional keyword argument 'msg' can be provided when assertRaises
           is used as a context object.

           The context manager keeps a reference to the exception as
           the 'exception' attribute. This allows you to inspect the
           exception after the assertion::

               with self.assertRaises(SomeException) as cm:
                   do_something()
               the_exception = cm.exception
               self.assertEqual(the_exception.error_code, 3)
        

---

    assertRaisesRegex(self, expected_exception, expected_regex, *args, **kwargs)

Asserts that the message in a raised exception matches a regex.

        Args:
            expected_exception: Exception class expected to be raised.
            expected_regex: Regex (re pattern object or string) expected
                    to be found in error message.
            args: Function to be called and extra positional args.
            kwargs: Extra kwargs.
            msg: Optional message used in case of failure. Can only be used
                    when assertRaisesRegex is used as a context manager.
        

---

    assertRaisesRegexp(*args, **kwargs)


---

    assertRegex(self, text, expected_regex, msg=None)

Fail the test unless the text matches the regular expression.

---

    assertRegexpMatches(*args, **kwargs)


---

    assertSequenceEqual(self, seq1, seq2, msg=None, seq_type=None)

An equality assertion for ordered sequences (like lists and tuples).

        For the purposes of this function, a valid ordered sequence type is one
        which can be indexed, has a length, and has an equality operator.

        Args:
            seq1: The first sequence to compare.
            seq2: The second sequence to compare.
            seq_type: The expected datatype of the sequences, or None if no
                    datatype should be enforced.
            msg: Optional message to use on failure instead of a list of
                    differences.
        

---

    assertSetEqual(self, set1, set2, msg=None)

A set-specific equality assertion.

        Args:
            set1: The first set to compare.
            set2: The second set to compare.
            msg: Optional message to use on failure instead of a list of
                    differences.

        assertSetEqual uses ducktyping to support different types of sets, and
        is optimized for sets specifically (parameters must support a
        difference method).
        

---

    assertTrue(self, expr, msg=None)

Check that the expression is true.

---

    assertTupleEqual(self, tuple1, tuple2, msg=None)

A tuple-specific equality assertion.

        Args:
            tuple1: The first tuple to compare.
            tuple2: The second tuple to compare.
            msg: Optional message to use on failure instead of a list of
                    differences.
        

---

    assertWarns(self, expected_warning, *args, **kwargs)

Fail unless a warning of class warnClass is triggered
           by the callable when invoked with specified positional and
           keyword arguments.  If a different type of warning is
           triggered, it will not be handled: depending on the other
           warning filtering rules in effect, it might be silenced, printed
           out, or raised as an exception.

           If called with the callable and arguments omitted, will return a
           context object used like this::

                with self.assertWarns(SomeWarning):
                    do_something()

           An optional keyword argument 'msg' can be provided when assertWarns
           is used as a context object.

           The context manager keeps a reference to the first matching
           warning as the 'warning' attribute; similarly, the 'filename'
           and 'lineno' attributes give you information about the line
           of Python code from which the warning was triggered.
           This allows you to inspect the warning after the assertion::

               with self.assertWarns(SomeWarning) as cm:
                   do_something()
               the_warning = cm.warning
               self.assertEqual(the_warning.some_attribute, 147)
        

---

    assertWarnsRegex(self, expected_warning, expected_regex, *args, **kwargs)

Asserts that the message in a triggered warning matches a regexp.
        Basic functioning is similar to assertWarns() with the addition
        that only warnings whose messages also match the regular expression
        are considered successful matches.

        Args:
            expected_warning: Warning class expected to be triggered.
            expected_regex: Regex (re pattern object or string) expected
                    to be found in error message.
            args: Function to be called and extra positional args.
            kwargs: Extra kwargs.
            msg: Optional message used in case of failure. Can only be used
                    when assertWarnsRegex is used as a context manager.
        

---

    assert_(*args, **kwargs)


---

    countTestCases(self)


---

    debug(self)

Run the test without collecting errors in a TestResult

---

    defaultTestResult(self)


---

    doCleanups(self)

Execute all cleanup functions. Normally called for you after
        tearDown.

---

    fail(self, msg=None)

Fail immediately, with the given message.

---

    failIf(*args, **kwargs)


---

    failIfAlmostEqual(*args, **kwargs)


---

    failIfEqual(*args, **kwargs)


---

    failUnless(*args, **kwargs)


---

    failUnlessAlmostEqual(*args, **kwargs)


---

    failUnlessEqual(*args, **kwargs)


---

    failUnlessRaises(*args, **kwargs)


---

    id(self)


---

    run(self, result=None)


---

    setUp(self)

Hook method for setting up the test fixture before exercising it.

---

    shortDescription(self)

Returns a one-line description of the test, or None if no
        description has been provided.

        The default implementation of this method returns the first line of
        the specified test method's docstring.
        

---

    skipTest(self, reason)

Skip this test.

---

    subTest(*args, **kwds)

Return a context manager that will return the enclosed block
        of code in a subtest identified by the optional message and
        keyword parameters.  A failure in the subtest marks the test
        case as failed but resumes execution at the end of the enclosed
        block, allowing further test code to be executed.
        

---

    tearDown(self)

Hook method for deconstructing the test fixture after testing it.

---

    test_two_wheel_2d_model(self)


---

    test_two_wheel_3d_model(self)


---

